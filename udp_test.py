from udp_multicast_tx_mac import MAC
from udp_multicast_tx_ip import IP
from udp_multicast_tx_udp import UDP
from udp_multicast_tx_streamer import Streamer
from pdm_capturer import PDMCapturerStereo
from cic_filter import CICFilter

from litex.gen import LiteXModule, ClockDomain, Signal, If
from litex.soc.integration.soc import SoCMini
from litex.soc.integration.builder import Builder
from litex.soc.cores.clock.lattice_ecp5 import ECP5PLL
from litex.soc.interconnect.stream import Endpoint, ClockDomainsRenamer, AsyncFIFO
from litex.build.generic_platform import Subsignal, Pins, IOStandard

from liteeth.common import convert_ip, eth_tty_tx_description
from liteeth.phy.ecp5rgmii import LiteEthPHYRGMII

from litex_boards.platforms.colorlight_i5 import Platform

import argparse
import os



_mp34dt01_pdm_microphones = [
    ("microphone", 0,
        Subsignal("data", Pins("N17")),
        Subsignal("clock", Pins("M18")),
        Subsignal("select_0", Pins("J20")),
        Subsignal("select_1", Pins("L18")),
        IOStandard("LVCMOS33"),
    ),
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

class PDMCapturer(LiteXModule):
    def __init__(self, microphone, data_width, clock_domain):
        self.source = Endpoint(eth_tty_tx_description(data_width))

        self.submodules.pdm_capturer_stereo = PDMCapturerStereo(
            oversample_ratio=25,
            left_sample_first_index=3,
            left_sample_last_index=15,
            left_sample_threshold=8,
            right_sample_first_index=16,
            right_sample_last_index=2,
            right_sample_threshold=8,
            clock_low_first_index=0,
            clock_high_first_index=13,
            valid_index=2,
        )

        number_of_pipelines = 25

        self.submodules.cic_filter = CICFilter(
            number_of_pipelines=number_of_pipelines,
            number_of_cic_stages=5,
            decimation_ratio=10,
            clock_domain=clock_domain,
        )

        self.comb += [
            microphone.select_0.eq(0),
            microphone.select_1.eq(1),

            microphone.clock.eq(self.pdm_capturer_stereo.microphone_clock),
            self.pdm_capturer_stereo.microphone_data.eq(microphone.data),
        ]

        counter = Signal(min=0, max=number_of_pipelines)

        pdm_left = Signal()
        pdm_right = Signal()

        self.sync += [
            If((self.pdm_capturer_stereo.pdm_data_valid),
                pdm_left.eq(self.pdm_capturer_stereo.pdm_data_left),
                pdm_right.eq(self.pdm_capturer_stereo.pdm_data_right),
            ),

            self.cic_filter.pdm_data_valid.eq(1),
            If((counter == 0),
                self.cic_filter.pdm_data.eq(pdm_left),
            ).Elif((counter == 1),
                self.cic_filter.pdm_data.eq(pdm_right),
            ).Else(
                self.cic_filter.pdm_data.eq(0),
            ),

            If((self.cic_filter.filtered_data_valid),
                self.source.data.eq(self.cic_filter.filtered_data),
                self.source.valid.eq(1),
            ).Else(
                self.source.valid.eq(0),
            ),

            If(counter == (number_of_pipelines - 1),
                counter.eq(0),
            ).Else(
                counter.eq(counter + 1),
            ),
        ]

class UDPTestSOC(SoCMini):
    def __init__(self, fpga_ip_address, fpga_mac_address, udp_multicast_ip_address, udp_multicast_ip_port, ethernet_phy_number):
        system_clock_frequency=60e6
        sample_clock_frequency=60e6

        platform = Platform(board="i5", revision="7.0", toolchain="trellis")
        platform.add_extension(_mp34dt01_pdm_microphones)

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

        microphone_0 = platform.request("microphone", 0)

        self.submodules.pdm_capturer = ClockDomainsRenamer("sample")(PDMCapturer(microphone_0, data_width, "sample"))

        self.fifo = ClockDomainsRenamer({"write": "sample", "read": "sys"})(AsyncFIFO(eth_tty_tx_description(data_width), depth=1024, buffered=True))

        self.comb += [
            self.pdm_capturer.source.connect(self.fifo.sink),
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
