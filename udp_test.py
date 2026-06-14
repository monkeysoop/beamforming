from udp_multicast_tx_mac import MAC
from udp_multicast_tx_ip import IP
from udp_multicast_tx_udp import UDP
from udp_multicast_tx_streamer import Streamer

from litex.gen import LiteXModule, ClockDomain, Signal, If, Cat
from litex.soc.integration.soc import SoCMini
from litex.soc.integration.builder import Builder
from litex.soc.cores.clock.lattice_ecp5 import ECP5PLL
from litex.soc.interconnect.stream import Endpoint, ClockDomainsRenamer, AsyncFIFO

from liteeth.common import convert_ip, eth_tty_tx_description
from liteeth.phy.ecp5rgmii import LiteEthPHYRGMII

from litex_boards.platforms.colorlight_i5 import Platform

import argparse
import os



class Counter(LiteXModule):
    def __init__(self, data_width):
        self.source = Endpoint(eth_tty_tx_description(data_width))

        counter = Signal(data_width)

        self.comb += [
            self.source.valid.eq(1),
            self.source.data.eq(Cat(*[counter[(data_width - i - 8):(data_width - i)] for i in range(0, data_width, 8)])),
        ]

        self.sync += [
            If((self.source.valid & self.source.ready),
                counter.eq(counter + 1),
            )
        ]

class _CRG(LiteXModule):
    def __init__(self, platform, system_clock_frequency, sample_clock_frequency):
        self.clock_domains.cd_sys = ClockDomain("sys")
        self.clock_domains.cd_sample = ClockDomain("sample")

        clk25 = platform.request("clk25")

        pll = ECP5PLL()
        self.submodules.pll = pll

        pll.register_clkin(clk25, 25e6)
        pll.create_clkout(self.cd_sys, system_clock_frequency, margin=0.0)
        pll.create_clkout(self.cd_sample, sample_clock_frequency, margin=0.0)

class UDPTestSOC(SoCMini):
    def __init__(self, fpga_ip_address, fpga_mac_address, udp_multicast_ip_address, udp_multicast_ip_port, ethernet_phy_number):
        system_clock_frequency=125e6
        sample_clock_frequency=5e6

        platform = Platform(board="i5", revision="7.0", toolchain="trellis")

        self.submodules.crg = _CRG(platform, system_clock_frequency, sample_clock_frequency)

        super().__init__(platform, system_clock_frequency)

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
            phy_clock_domain="eth_tx",
            core_clock_domain="sys",
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
            data_width=data_width,
            internal_clock_domain="sys",
        )

        self.submodules.streamer = Streamer(
            udp=self.udp,
            udp_multicast_ip_address=convert_ip(udp_multicast_ip_address),
            udp_multicast_ip_port=udp_multicast_ip_port,
            data_width=data_width,
            internal_clock_domain="sys",
            fifo_depth=256,
        )

        self.submodules.counter = ClockDomainsRenamer("sample")(Counter(data_width))

        self.fifo = ClockDomainsRenamer({"write": "sample", "read": "sys"})(AsyncFIFO(eth_tty_tx_description(data_width), depth=None, buffered=False))

        self.comb += [
            self.counter.source.connect(self.fifo.sink),
            self.fifo.source.connect(self.streamer.sink),
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
