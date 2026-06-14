from litex.gen import LiteXModule, If, FSM, Signal, NextValue, NextState, ResetInserter
from litex.soc.interconnect.stream import Endpoint, SyncFIFO

from liteeth.common import eth_udp_user_description, eth_tty_tx_description



class StreamerTX(LiteXModule):
    def __init__(self, udp_multicast_ip_address, udp_multicast_ip_port, data_width, fifo_depth):
        self.sink = Endpoint(eth_tty_tx_description(data_width))
        self.source = Endpoint(eth_udp_user_description(data_width))

        self.fifo = SyncFIFO([("data", data_width)], fifo_depth, buffered=True)
        self.comb += [
            self.sink.connect(self.fifo.sink),
        ]

        level   = Signal(max=(fifo_depth + 1))
        counter = Signal(max=(fifo_depth + 1))

        self.fsm = ResetInserter()(FSM(reset_state="IDLE"))
        self.fsm.act("IDLE",
            NextValue(counter, 0),
            If((self.fifo.sink.valid & self.fifo.sink.ready & self.fifo.sink.last),
                NextValue(level, self.fifo.level + 1),
                NextState("SEND"),
            ),
            If((~self.fifo.sink.ready),
                NextValue(level, fifo_depth),
                NextState("SEND"),
            ),
        )
        self.fsm.act("SEND",
            self.source.valid.eq(1),
            self.source.last.eq(counter == (level - 1)),
            self.source.src_port.eq(udp_multicast_ip_port),
            self.source.dst_port.eq(udp_multicast_ip_port),
            self.source.ip_address.eq(udp_multicast_ip_address),
            self.source.length.eq(level * (data_width // 8)),
            self.source.data.eq(self.fifo.source.data),
            If((self.source.last),
                self.source.last_be.eq(0b1 << ((data_width // 8) - 1)),
            ),
            If((self.source.ready),
                self.fifo.source.ready.eq(1),
                NextValue(counter, counter + 1),
                If((counter == (level - 1)),
                    NextState("IDLE"),
                ),
            ),
        )

class Streamer(LiteXModule):
    def __init__(self, udp, udp_multicast_ip_address, udp_multicast_ip_port, data_width, internal_clock_domain, fifo_depth):
        self.streamer_tx = StreamerTX(udp_multicast_ip_address, udp_multicast_ip_port, data_width, fifo_depth)
        self.port = udp.udp_crossbar.get_port(udp_ip_port=udp_multicast_ip_port, data_width=data_width, clock_domain=internal_clock_domain)
        
        self.comb += [
            self.streamer_tx.source.connect(self.port.sink),
        ]

        self.sink = self.streamer_tx.sink
