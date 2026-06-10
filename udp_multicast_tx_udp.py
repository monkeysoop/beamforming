from litex.gen import LiteXModule, If, FSM, NextState
from litex.soc.interconnect.stream import Endpoint, ClockDomainCrossing, StrideConverter

from liteeth.common import udp_protocol, udp_header, eth_udp_user_description, eth_ipv4_user_description, eth_udp_description
from liteeth.crossbar import LiteEthCrossbar
from liteeth.packet import Packetizer



class UDPMasterPort:
    def __init__(self, data_width):
        self.source = Endpoint(eth_udp_user_description(data_width))
        self.sink = Endpoint(eth_udp_user_description(data_width))


class UDPSlavePort:
    def __init__(self, data_width):
        self.sink = Endpoint(eth_udp_user_description(data_width))
        self.source = Endpoint(eth_udp_user_description(data_width))

class UDPCrossbar(LiteEthCrossbar):
    def __init__(self, data_width):
        self.data_width = data_width
        LiteEthCrossbar.__init__(self, UDPMasterPort, "dst_port", dw=data_width)

    def get_port(self, udp_ip_port, data_width, clock_domain):
        if (udp_ip_port in self.users.keys()):
            raise ValueError("Error, udp ip port {0:#x} already taken".format(udp_ip_port))

        port = UDPSlavePort(data_width)
        internal_port = UDPSlavePort(self.data_width)

        self.users[udp_ip_port] = internal_port

        self.clock_domain_crossing = ClockDomainCrossing(layout=eth_udp_user_description(data_width), cd_from=clock_domain, cd_to="sys")
        self.stride_converter = StrideConverter(description_from=eth_udp_user_description(data_width), description_to=eth_udp_user_description(self.data_width))

        self.comb += [
            port.sink.connect(self.clock_domain_crossing.sink),
            self.clock_domain_crossing.source.connect(self.stride_converter.sink),
            self.stride_converter.source.connect(internal_port.sink),
        ]

        return port

class UDPTX(LiteXModule):
    def __init__(self, ip_address, data_width):
        self.sink = Endpoint(eth_udp_user_description(data_width))
        self.source = Endpoint(eth_ipv4_user_description(data_width))

        self.packetizer = Packetizer(eth_udp_description(data_width), eth_ipv4_user_description(data_width), udp_header)

        self.comb += [
            self.sink.connect(
                self.packetizer.sink,
                keep={
                    "valid",
                    "ready",
                    "last",
                    "last_be",
                    "src_port",
                    "dst_port",
                    "data",
                },
            ),
            self.packetizer.sink.length.eq(self.sink.length + udp_header.length),
            self.packetizer.sink.checksum.eq(0),
        ]

        self.fsm = FSM(reset_state="IDLE")
        self.fsm.act(
            "IDLE",
            If((self.packetizer.source.valid),
                NextState("SEND"),
            ),
        )
        self.fsm.act(
            "SEND",
            self.packetizer.source.connect(self.source),
            self.source.length.eq(self.packetizer.sink.length),
            self.source.protocol.eq(udp_protocol),
            self.source.ip_address.eq(self.sink.ip_address),
            If((self.source.valid & self.source.ready & self.source.last),
                NextState("IDLE"),
            ),
        )

class UDP(LiteXModule):
    def __init__(self, ip, ip_address, data_width):
        self.udp_crossbar = UDPCrossbar(data_width)
        self.udp_tx = UDPTX(ip_address, data_width)
        ip_port = ip.ip_crossbar.get_port(udp_protocol, data_width)

        self.comb += [
            self.udp_crossbar.master.source.connect(self.udp_tx.sink),
            self.udp_tx.source.connect(ip_port.sink),
        ]
