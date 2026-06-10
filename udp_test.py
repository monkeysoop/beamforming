from udp_multicast_tx_mac import MAC
from udp_multicast_tx_ip import IP
from udp_multicast_tx_udp import UDP

from litex.gen import LiteXModule, ClockDomain, Signal, If
from litex.soc.integration.soc import SoCMini
from litex.soc.integration.builder import Builder
from litex.soc.cores.clock.lattice_ecp5 import ECP5PLL

from liteeth.common import convert_ip
from liteeth.phy.ecp5rgmii import LiteEthPHYRGMII

from litex_boards.platforms.colorlight_i5 import Platform

import argparse
import os



class _CRG(LiteXModule):
    def __init__(self, platform, sys_clk_freq):
        self.clock_domains.cd_sys = ClockDomain("sys")

        clk25 = platform.request("clk25")

        pll = ECP5PLL()
        self.submodules.pll = pll

        pll.register_clkin(clk25, 25e6)
        pll.create_clkout(self.cd_sys, sys_clk_freq, margin=0.0)

class UDPTestSOC(SoCMini):
    def __init__(self, fpga_ip_address, fpga_mac_address, udp_multicast_ip_address, udp_multicast_ip_port, ethernet_phy_number):
        sys_clk_freq=125e6

        platform = Platform(board="i5", revision="7.0", toolchain="trellis")

        self.submodules.crg = _CRG(platform, sys_clk_freq)

        super().__init__(platform, sys_clk_freq)

        led_wire = platform.request("user_led_n", 0)

        self.submodules.ethphy = LiteEthPHYRGMII(
            clock_pads =self.platform.request("eth_clocks", ethernet_phy_number),
            pads=self.platform.request("eth", ethernet_phy_number),
            tx_delay=0e-9,
            rx_delay=2e-9,
        )

        data_width = 32

        self.submodules.mac = MAC(
            phy=self.ethphy,
            data_width=data_width,
            with_preamble_crc=True,
            with_sys_datapath=True,
        )

        self.submodules.ip = IP(
            mac=self.mac,
            mac_address=fpga_mac_address,
            ip_address=convert_ip(fpga_ip_address),
            data_width=data_width,
        )

        self.submodules.udp = UDP(
            ip=self.ip,
            ip_address=None,
            data_width=data_width,
        )

        port = self.udp.udp_crossbar.get_port(
            udp_multicast_ip_port,
            data_width=data_width,
            clock_domain="sys",
        )

        counter = Signal(32)
        word_count = Signal(8)

        PACKET_WORDS = 200

        self.comb += [
            port.sink.valid.eq(1),
            port.sink.last.eq(word_count == (PACKET_WORDS - 1)),

            port.sink.payload.error.eq(0),

            port.sink.param.src_port.eq(udp_multicast_ip_port),
            port.sink.param.dst_port.eq(udp_multicast_ip_port),
            port.sink.param.ip_address.eq(convert_ip(udp_multicast_ip_address)),
            port.sink.param.length.eq(PACKET_WORDS * (data_width // 8)),

        ]

        self.comb += [
            If((word_count == (PACKET_WORDS - 1)),
                port.sink.payload.last_be.eq(0b1000),
            ),
        ]
        
        self.sync += [
            If((port.sink.valid & port.sink.ready),
                counter.eq(counter + 1),

                If((word_count == (PACKET_WORDS - 1)),
                    word_count.eq(0),
                ).Else(
                    word_count.eq(word_count + 1),
                ),

                port.sink.payload.data.eq(0xABCD0000 | word_count),
            )
        ]

def main():
    parser = argparse.ArgumentParser(description="UDP test streamer program for a colorlight i5 fpga")
    parser.add_argument("--build", action="store_true", help="Build bitstream")
    parser.add_argument("--load", action="store_true", help="Load bitstream")
    parser.add_argument("--fpga-ip-address",  default="192.168.1.2",   help="Ethernet IP address of the fpga board (default: 192.168.1.2).")
    parser.add_argument("--fpga-mac-address", default="0x0123456789AB", help="Ethernet MAC address of the fpga board (default: 0x0123456789AB).")
    parser.add_argument("--udp-multicast-ip-address",  default="239.1.2.3",   help="Ethernet IP address of the (target pc's) udp multicast group (default: 239.1.2.3).")
    parser.add_argument("--udp-multicast-ip-port",  default="1234",   help="Ethernet IP port of the (target pc's) udp multicast group (default: 1234).")
    parser.add_argument("--ethernet-phy-number",  default="0",   help="Ethernet PHY chip number on the fpga (default: 0).")
    args = parser.parse_args()

    soc = UDPTestSOC(
        args.fpga_ip_address,
        int(args.fpga_mac_address, 16),
        args.udp_multicast_ip_address,
        int(args.udp_multicast_ip_port),
        int(args.ethernet_phy_number),
    )

    builder = Builder(soc, output_dir="build", csr_csv="csr.csv")
    builder.build(build_name="udp_test_soc", run=args.build)

    if args.load:
        prog = soc.platform.create_programmer()
        prog.load_bitstream(os.path.join(builder.gateware_dir, soc.build_name + ".bit"))

if __name__ == "__main__":
    main()
