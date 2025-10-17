# tests/conftest.py
import pytest
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, SKOS

@pytest.fixture(scope="session")
def toy_graph():
    g = Graph()
    IEC    = Namespace("http://iec.ch/TC57/2012/CIM-schema-cim17#")
    ARCHI  = Namespace("http://archimate.example/")
    ALL    = Namespace("https://vocabs.alliander.com/terms/")
    LIDO   = Namespace("http://lido.example/")
    ELIT   = Namespace("http://data.europa.eu/eli/terms/")

    g.bind("iec", IEC); g.bind("archi", ARCHI); g.bind("all", ALL)
    g.bind("lido", LIDO); g.bind("elit", ELIT); g.bind("skos", SKOS)

    def concept(ns, local, label_en, definition_en=None, label_nl=None):
        s = URIRef(ns[local])
        g.add((s, RDF.type, SKOS.Concept))
        g.add((s, SKOS.prefLabel, Literal(label_en, lang="en")))
        if label_nl:
            g.add((s, SKOS.prefLabel, Literal(label_nl, lang="nl")))
        if definition_en is not None:
            g.add((s, SKOS.definition, Literal(definition_en, lang="en")))
        return s

    # Ambiguous: Service
    concept(IEC,   "ServiceSupply",             "Service", "Electrical service connection and feeder")
    concept(ALL,   "CustomerService",           "Service", "Business service provided to customers")
    concept(ARCHI, "TelemetryIngestionService", "Service", "Application service exposing ingestion API")
    concept(LIDO,  "RegulatedService",          "Service", "Regulatory service per directive")

    # Ambiguous: Model
    concept(IEC,   "PowerSystemModel",          "Model",   "Model of electrical network and equipment")
    concept(ARCHI, "LogicalDataModel",          "Model",   "Logical data model of telemetry schema")

    # Ambiguous: Network
    concept(IEC,   "TransmissionNetwork",       "Network", "High voltage transmission network")
    concept(ALL,   "PremisesAreaNetwork",       "Network", "Corporate data network on premises")

    # Ambiguous: Flow
    concept(IEC,   "PowerFlow",                 "Flow",    "Active and reactive power flow")
    concept(ARCHI, "TelemetryDataFlow",         "Flow",    "Data flow from meters to ingestion")

    # Ambiguous: Phase
    concept(IEC,   "ElectricalPhase",           "Phase",   "Electrical phase angle and sequence")
    concept(ARCHI, "TOGAFPhaseB",               "Phase",   "TOGAF Phase B: Business Architecture")

    # Non-ambiguous: Current
    concept(IEC,   "ElectricalCurrent",         "Current", "Rate of electric charge flow")

    # Edge: no definition
    concept(ELIT,  "ComplianceRequirement",     "Requirement", None)

    return g
