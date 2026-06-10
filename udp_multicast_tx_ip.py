from litex.gen import LiteXModule, If, FSM, NextState, NextValue, Cat, Signal
from litex.soc.interconnect.stream import Endpoint, Buffer

from liteeth.common import ipv4_header, mcast_oui, ethernet_type_ip, eth_ipv4_user_description, eth_ipv4_description, eth_mac_description
from liteeth.crossbar import LiteEthCrossbar
from liteeth.packet import Packetizer
from liteeth.core.ip import LiteEthIPV4Checksum



class IPMasterPort:
    def __init__(self, data_width):
        self.source = Endpoint(eth_ipv4_user_description(data_width))
        self.sink = Endpoint(eth_ipv4_user_description(data_width))

class IPSlavePort:
    def __init__(self, data_width):
        self.sink = Endpoint(eth_ipv4_user_description(data_width))
        self.source = Endpoint(eth_ipv4_user_description(data_width))

class IPCrossbar(LiteEthCrossbar):
    def __init__(self, data_width):
        LiteEthCrossbar.__init__(self, IPMasterPort, "protocol", data_width)

    def get_port(self, protocol, data_width):
        if (protocol in self.users.keys()):
            raise ValueError("Error, ip protocol {0:#x} already taken".format(protocol))

        port = IPSlavePort(data_width)

        self.users[protocol] = port

        return port

class IPTX(LiteXModule):
    def __init__(self, mac_address, ip_address, data_width, with_buffer=True):
        self.source = Endpoint(eth_mac_description(data_width))
        self.sink = Endpoint(eth_ipv4_user_description(data_width))

        if with_buffer:
            self.buffer = Buffer(eth_ipv4_user_description(data_width))
            self.comb += self.sink.connect(self.buffer.sink)
            local_sink = self.buffer.source
        else:
            local_sink = self.sink

        self.checksum = LiteEthIPV4Checksum(skip_checksum=True)
        self.comb += [
            self.checksum.ce.eq(local_sink.valid),
            self.checksum.reset.eq(self.source.valid & self.source.last & self.source.ready),
        ]

        self.packetizer = Packetizer(eth_ipv4_description(data_width), eth_mac_description(data_width), ipv4_header)
        self.comb += [
            local_sink.connect(
                self.packetizer.sink,
                keep={
                    "last",
                    "last_be",
                    "protocol",
                    "data",
                },
            ),
            self.packetizer.sink.valid.eq(local_sink.valid & self.checksum.done),
            local_sink.ready.eq(self.packetizer.sink.ready & self.checksum.done),
            self.packetizer.sink.target_ip.eq(local_sink.ip_address),
            self.packetizer.sink.total_length.eq(ipv4_header.length + local_sink.length),
            self.packetizer.sink.version.eq(0x4),
            self.packetizer.sink.ihl.eq(ipv4_header.length // 4),
            self.packetizer.sink.identification.eq(0),
            self.packetizer.sink.ttl.eq(0x80),
            self.packetizer.sink.sender_ip.eq(ip_address),
            self.checksum.header.eq(self.packetizer.header),
            self.packetizer.sink.checksum.eq(self.checksum.value),
        ]

        target_mac = Signal(48, reset_less=True)

        self.fsm = FSM(reset_state="IDLE")
        self.fsm.act(
            "IDLE",
            If((self.packetizer.source.valid),
                NextValue(target_mac, Cat(local_sink.ip_address[:23], 0, mcast_oui)),
                NextState("SEND"),
            ),
        )
        self.fsm.act(
            "SEND",
            self.packetizer.source.connect(self.source),
            self.source.ethernet_type.eq(ethernet_type_ip),
            self.source.target_mac.eq(target_mac),
            self.source.sender_mac.eq(mac_address),
            If((self.source.valid & self.source.last & self.source.ready),
                NextState("IDLE"),
            ),
        )
        self.fsm.act(
            "DROP",
            self.packetizer.source.ready.eq(1),
            If((self.packetizer.source.valid & self.packetizer.source.last & self.packetizer.source.ready),
                NextState("IDLE"),
            ),
        )

class IP(LiteXModule):
    def __init__(self, mac, mac_address, ip_address, data_width):
        self.ip_crossbar = IPCrossbar(data_width)
        self.ip_tx = IPTX(mac_address, ip_address, data_width=data_width)
        mac_port = mac.mac_crossbar.get_port(ethernet_type_ip, data_width)

        self.comb += [
            self.ip_crossbar.master.source.connect(self.ip_tx.sink),
            self.ip_tx.source.connect(mac_port.sink),
        ]
