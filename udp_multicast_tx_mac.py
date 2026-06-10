from litex.gen import LiteXModule
from litex.soc.interconnect.csr import CSRStatus
from litex.soc.interconnect.stream import Endpoint, ClockDomainCrossing, ClockDomainsRenamer, StrideConverter, BufferizeEndpoints, Pipeline, DIR_SINK

from liteeth.common import eth_phy_description, eth_mac_description, mac_header, eth_min_frame_length, eth_fcs_length
from liteeth.crossbar import LiteEthCrossbar
from liteeth.packet import Packetizer
from liteeth.mac.last_be import LiteEthMACTXLastBE
from liteeth.mac.padding import LiteEthMACPaddingInserter
from liteeth.mac.crc import LiteEthMACCRC32Inserter
from liteeth.mac.preamble import LiteEthMACPreambleInserter
from liteeth.mac.gap import LiteEthMACGap



class MACMasterPort:
    def __init__(self, data_width):
        self.source = Endpoint(eth_mac_description(data_width))
        self.sink = Endpoint(eth_mac_description(data_width))


class MACSlavePort:
    def __init__(self, data_width):
        self.sink = Endpoint(eth_mac_description(data_width))
        self.source = Endpoint(eth_mac_description(data_width))

class MACCrossbar(LiteEthCrossbar):
    def __init__(self, data_width):
        LiteEthCrossbar.__init__(self, MACMasterPort, "ethernet_type", data_width)

    def get_port(self, ethernet_type, data_width):
        if ethernet_type in self.users.keys():
            raise ValueError("Ethernet type {0:#x} already assigned".format(ethernet_type))

        port = MACSlavePort(data_width)

        self.users[ethernet_type] = port

        return port

class TXDatapath(LiteXModule):
    def __init__(self):
        self.pipeline = []

    def add_clock_domain_crossing(self, core_data_width, depth=32, buffered=False):
        clock_domain_crossing = ClockDomainCrossing(eth_phy_description(core_data_width), cd_from="sys", cd_to="eth_tx", depth=depth, buffered=buffered)
        self.submodules += clock_domain_crossing
        self.pipeline.append(clock_domain_crossing)

    def add_stride_converter(self, core_data_width, phy_data_width):
        stride_converter = ClockDomainsRenamer("eth_tx")(StrideConverter(description_from=eth_phy_description(core_data_width), description_to=eth_phy_description(phy_data_width)))
        self.submodules += stride_converter
        self.pipeline.append(stride_converter)

    def add_last_be(self, phy_data_width):
        last_be = ClockDomainsRenamer("eth_tx")(LiteEthMACTXLastBE(phy_data_width))
        self.submodules += last_be
        self.pipeline.append(last_be)

    def add_padding(self, datapath_data_width, clock_domain):
        padding = ClockDomainsRenamer(clock_domain)(LiteEthMACPaddingInserter(datapath_data_width, (eth_min_frame_length - eth_fcs_length)))
        self.submodules += padding
        self.pipeline.append(padding)

    def add_crc32(self, datapath_data_width, clock_domain):
        crc32 = LiteEthMACCRC32Inserter(eth_phy_description(datapath_data_width))
        crc32 = BufferizeEndpoints({"sink": DIR_SINK})(crc32)
        crc32 = ClockDomainsRenamer(clock_domain)(crc32)
        self.submodules += crc32
        self.pipeline.append(crc32)

    def add_preamble(self, datapath_data_width, clock_domain):
        preamble = ClockDomainsRenamer(clock_domain)(LiteEthMACPreambleInserter(datapath_data_width))
        self.submodules += preamble
        self.pipeline.append(preamble)

    def add_gap(self, phy_data_width):
        gap = ClockDomainsRenamer("eth_tx")(LiteEthMACGap(phy_data_width))
        self.submodules += gap
        self.pipeline.append(gap)

    def do_finalize(self):
        self.submodules += Pipeline(*self.pipeline)

class MACCore(LiteXModule):
    def __init__(self, phy, data_width, with_sys_datapath=False, with_preamble_crc=True, with_padding=True):
        self.sink = Endpoint(eth_phy_description(data_width))

        if (data_width < phy.dw):
            raise ValueError("Error, mac core data width: {} must be larger than PHY data width: {}".format(data_width, phy.dw))

        clock_domain = ("sys" if (with_sys_datapath) else "eth_tx")
        datapath_data_width = (data_width if (with_sys_datapath) else phy.dw)

        if (hasattr(phy, "with_preamble_crc")):
            with_preamble_crc = phy.with_preamble_crc
        if (hasattr(phy, "with_padding")):
            with_padding = phy.with_padding

        self.datapath = TXDatapath()
        self.datapath.pipeline.append(self.sink)

        if (not with_sys_datapath):
            self.datapath.add_clock_domain_crossing(data_width)
            if (data_width != phy.dw):
                self.datapath.add_stride_converter(data_width, phy.dw)
            if (data_width != 8):
                self.datapath.add_last_be(phy.dw)

        if (with_padding):
            self.datapath.add_padding(datapath_data_width, clock_domain)

        if (with_preamble_crc):
            self.datapath.add_crc32(datapath_data_width, clock_domain)
            self.datapath.add_preamble(datapath_data_width, clock_domain)

        if (with_sys_datapath):
            self.datapath.add_clock_domain_crossing(data_width)
            if (data_width != phy.dw):
                self.datapath.add_stride_converter(data_width, phy.dw)
            if (data_width != 8):
                self.datapath.add_last_be(phy.dw)

        if (not getattr(phy, "integrated_ifg_inserter", False)):
            self.datapath.add_gap(phy.dw)

        self.datapath.pipeline.append(phy)

class MAC(LiteXModule):
    def __init__(self, phy, data_width, with_preamble_crc=True, with_sys_datapath=False):
        self.mac_core = MACCore(phy, data_width, with_sys_datapath=with_sys_datapath, with_preamble_crc=with_preamble_crc)
        self.mac_crossbar = MACCrossbar(data_width)
        self.mac_packetizer = Packetizer(eth_mac_description(data_width), eth_phy_description(data_width), mac_header)

        self.comb += [
            self.mac_crossbar.master.source.connect(self.mac_packetizer.sink),
            self.mac_packetizer.source.connect(self.mac_core.sink),
        ]
