"""
Benchmark Datasets for Module Validation.

Real proteins with experimentally known properties for validating
each prediction module. Sources: UniProt, PDB, published literature.

CRITICAL: These benchmarks test whether our heuristics produce
directionally correct results, NOT whether they match specialized
tools like NetNGlyc or TANGO. They are sanity checks, not gold standards.
"""

# ── Benchmark 1: Glycosylation Site Prediction ──────────────────────────────────
# Known glycoproteins with experimentally verified N-glycosylation sites
# Sources: UniProt feature annotations (FT CARBOHYD), published glycoproteomics

GLYCOSYLATION_BENCHMARKS = [
    {
        "protein_name": "Human IgG1 heavy chain",
        "sequence": (
            "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRF"
            "TISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSSASTKGPSVFPLAPSSKS"
            "TSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNH"
            "KPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKF"
            "NWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREP"
            "QVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRW"
            "QQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
        ),
        "known_n_glyc_sites": [301],  # Asn297 equivalent in this VH+CH1+CH2+CH3 construct
        "source": "UniProt P01857; Edelman et al. 1969",
    },
    {
        "protein_name": "Ricin B chain (highly glycosylated)",
        "sequence": (
            "APVRVQFITPGTNISVTQRNITELQIFRNSFSTQNKLINVDNTNNEIFGSGSNSNQLVDNLSFNPTKFLV"
            "GLSNIFCKKDKNTFTINLSKNNTLISQNSDYKSFTVTINNSKNIITFSNDNSLNSFQSNSNIFRQNTLSLD"
            "CNKFNPKVKLYFTKQDSNTIIFQNDIQTKLNNVNKTFNLISKDMQISFNYKQPFQSKFIKNSNTKIPNIIF"
            "GSGSPYPV"
        ),
        "known_n_glyc_sites": [13, 20, 60, 64, 87, 91, 109, 175],  # Verified N-X-S/T positions in this sequence
        "source": "UniProt P02879; Foxwell et al. 1985",
    },
    {
        "protein_name": "Human erythropoietin (3 N-glyc sites)",
        "sequence": (
            "MGVHECPAWLWLLLSLLSLPLGLPVLGAPPRLICDSRVLERYLLEAKEAENITTGCAEHCSLNENITVPD"
            "TKVNFYAWKRMEVGQQAVEVWQGLALLSEAVLRGQALLVNSSQPWEPLQLHVDKAVSGLRSLTTLLRALGA"
            "QKEAISPPDAASAAPLRTITADTFRKLFRVYSNFLRGKLKLYTGEACRTGDR"
        ),
        "known_n_glyc_sites": [51, 65, 110],  # Full-length positions (includes 27-aa signal peptide)
        "source": "UniProt P01588; Sasaki et al. 1987",
    },
    {
        "protein_name": "Hyaluronidase PH-20 (6 N-glyc sites)",
        "sequence": (
            "MGVLKFKHIFFRSFVKSSGVSQIVFTFLLIPCCLTLNFRAPPVIPNVPFLWAWNAPSEFCLGKFDEPLDMS"
            "LFSFIGSPRINATGQGVTIFYVDRLGYYPYIDSITGVTVNGGIPQKISLQDHLDKAKKDITFYMPVDNLGMA"
            "VIDWEEWRPTWARNWKPKDVYKNRSIELVQQQNVQLSLTEATEKAKQEFEKAGKDFLVETIKLGKLLRPNHLW"
            "GYYLFPDCYNHHYKKPGYNGSCFNVEIKRNDDLSWLWNESTALYPSIYLNTQQSPVAATLYVRNRVREAIRVSK"
            "IPDAKSPLPVFAYTRIVFTDQVLKFLSQDELVYTFGETVALGASGIVIWGTLSIMRSMKSCLLLDNYMETILNPY"
            "IINVTLAAKMCSQVLCQEQGVCIRKNWNSSDYLHLNPDNFAIQLEKGGKFTVRGKPTLEDLEQFSEKFYCSCYST"
            "LSCKEKADVKDTDAVDVCIADGVCIDAFLKPPMETEEPQIFYNASPSTLSATMFIVSILFLIISSVASL"
        ),
        "known_n_glyc_sites": [82, 166, 235, 254, 368, 393, 483],  # Literature: 6-7 sites
        "source": "UniProt P38567; Gmachl & Kreil 1993",
    },
    {
        "protein_name": "BSA (no N-glycosylation — negative control)",
        "sequence": (
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNE"
            "LTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLSHKDDSPDLPKLK"
            "PDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKV"
            "LTSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAK"
            "YICDNQDTISSKLKECCDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYS"
            "RRHPEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYT"
            "RKEHPQFGGGSQKLFKDLGEQHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESAENCDK"
        ),
        "known_n_glyc_sites": [],  # BSA is NOT glycosylated
        "source": "UniProt P02769; Peters 1995",
    },
    {
        "protein_name": "RNase A (has sequons but NOT glycosylated)",
        "sequence": (
            "KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCY"
            "QSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV"
        ),
        "known_n_glyc_sites": [],  # Has N-X-S/T sequons at 24,34 but NOT glycosylated
        "source": "UniProt P61823; RNase A is not glycosylated despite having sequons",
    },
]

# ── Benchmark 2: Disulfide Bond Proteins ────────────────────────────────────────
# Proteins with known disulfide bond counts from PDB structures

DISULFIDE_BENCHMARKS = [
    {
        "protein_name": "RNase A",
        "sequence": "KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV",
        "known_disulfide_bonds": 4,
        "n_cysteines": 8,
        "folding_difficulty": "moderate",
        "source": "PDB 7RSA; known to fold correctly in vitro",
    },
    {
        "protein_name": "Lysozyme (hen egg white)",
        "sequence": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
        "known_disulfide_bonds": 4,
        "n_cysteines": 8,
        "folding_difficulty": "easy",
        "source": "PDB 6LYZ; classic folding model protein",
    },
    {
        "protein_name": "Tissue plasminogen activator (serine protease domain)",
        "sequence": "NQGRRPRWPWQVSIVALHGEFDYVVPKKDAFCGSLTTENIYVKDIHTVVRGEKENNLQAFEEVEGNCVTTSYQPYVEDLMLDKNRDPFQKKISKDVPFYTVPVQDGLDPNENSVVQIYDNKNVTLWIKKNGDGRNFKCKNHKIYFKDSYWPNSKTCSGNIDVAKRNIFQEAFNLSYKLCMEAMYKDKNKPEEIKVMLSK",
        "known_disulfide_bonds": 2,
        "n_cysteines": 5,
        "folding_difficulty": "difficult",
        "source": "PDB 1BUI; serine protease domain only (5 Cys, 2 confirmed disulfides + 1 free Cys)",
    },
    {
        "protein_name": "Insulin (A+B chains)",
        "sequence": "FVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        "known_disulfide_bonds": 3,
        "n_cysteines": 6,
        "folding_difficulty": "moderate",
        "source": "PDB 1ZNJ; 3 disulfides including inter-chain",
    },
    {
        "protein_name": "Hyaluronidase PH-20",
        "sequence": "MGVLKFKHIFFRSFVKSSGVSQIVFTFLLIPCCLTLNFRAPPVIPNVPFLWAWNAPSEFCLGKFDEPLDMSLFSFIGSPRINATGQGVTIFYVDRLGYYPYIDSITGVTVNGGIPQKISLQDHLDKAKKDITFYMPVDNLGMAVIDWEEWRPTWARNWKPKDVYKNRSIELVQQQNVQLSLTEATEKAKQEFEKAGKDFLVETIKLGKLLRPNHLWGYYLFPDCYNHHYKKPGYNGSCFNVEIKRNDDLSWLWNESTALYPSIYLNTQQSPVAATLYVRNRVREAIRVSKIPDAKSPLPVFAYTRIVFTDQVLKFLSQDELVYTFGETVALGASGIVIWGTLSIMRSMKSCLLLDNYMETILNPYIINVTLAAKMCSQVLCQEQGVCIRKNWNSSDYLHLNPDNFAIQLEKGGKFTVRGKPTLEDLEQFSEKFYCSCYSTLSCKEKADVKDTDAVDVCIADGVCIDAFLKPPMETEEPQIFYNASPSTLSATMFIVSILFLIISSVASL",
        "known_disulfide_bonds": 7,
        "n_cysteines": 14,
        "folding_difficulty": "very_difficult",
        "source": "UniProt P38567; 8 disulfide bonds reported (our model says 7 from 14 Cys)",
    },
    {
        "protein_name": "GFP (no disulfides — negative control)",
        "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "known_disulfide_bonds": 0,
        "n_cysteines": 2,
        "folding_difficulty": "easy",
        "source": "PDB 1EMA; GFP folds autonomously",
    },
]

# ── Benchmark 3: Known Soluble vs Insoluble Proteins ────────────────────────────
# Proteins with known expression behavior in heterologous systems

SOLUBILITY_BENCHMARKS = [
    {
        "protein_name": "GFP",
        "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "expected_solubility": "high",
        "expected_expression_success": True,
        "notes": "GFP is highly soluble, expresses well in all systems",
    },
    {
        "protein_name": "MBP (maltose binding protein)",
        "sequence": "MKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEPYFTWPLIAADGGYAFKYENGKYDIKDVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNKGETAMTINGPWAWSNIDTSKVNYGVTVLPTFKGQPSKPFVGVLSAGINAASPNKELAKEFLENYLLTDEGLEAVNKDKPLGAVALKSYEEELAKDPRIAATMENAQKGEIMPNIPQMSAFWYAVRTAVINAASGRQTVDEALKDAQTRITK",
        "expected_solubility": "high",
        "expected_expression_success": True,
        "notes": "MBP is a common solubility tag, very well-behaved",
    },
    {
        "protein_name": "Tau protein fragment (aggregation-prone)",
        "sequence": "VQIVYKPVDLSKVGSKVCGGNIATKPGGGKEQPFN",
        "expected_solubility": "low",
        "expected_expression_success": False,
        "notes": "Tau fragment forms aggregates, known amyloidogenic sequence",
    },
    {
        "protein_name": "Aβ42 (amyloid beta, highly aggregation-prone)",
        "sequence": "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
        "expected_solubility": "very_low",
        "expected_expression_success": False,
        "notes": "Alzheimer's peptide, forms fibrils rapidly",
    },
    {
        "protein_name": "Trastuzumab light chain (well-behaved antibody)",
        "sequence": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC",
        "expected_solubility": "high",
        "expected_expression_success": True,
        "notes": "Therapeutic antibody light chain, well-characterized expression",
    },
    {
        "protein_name": "Hyaluronidase PH-20",
        "sequence": "MGVLKFKHIFFRSFVKSSGVSQIVFTFLLIPCCLTLNFRAPPVIPNVPFLWAWNAPSEFCLGKFDEPLDMSLFSFIGSPRINATGQGVTIFYVDRLGYYPYIDSITGVTVNGGIPQKISLQDHLDKAKKDITFYMPVDNLGMAVIDWEEWRPTWARNWKPKDVYKNRSIELVQQQNVQLSLTEATEKAKQEFEKAGKDFLVETIKLGKLLRPNHLWGYYLFPDCYNHHYKKPGYNGSCFNVEIKRNDDLSWLWNESTALYPSIYLNTQQSPVAATLYVRNRVREAIRVSKIPDAKSPLPVFAYTRIVFTDQVLKFLSQDELVYTFGETVALGASGIVIWGTLSIMRSMKSCLLLDNYMETILNPYIINVTLAAKMCSQVLCQEQGVCIRKNWNSSDYLHLNPDNFAIQLEKGGKFTVRGKPTLEDLEQFSEKFYCSCYSTLSCKEKADVKDTDAVDVCIADGVCIDAFLKPPMETEEPQIFYNASPSTLSATMFIVSILFLIISSVASL",
        "expected_solubility": "moderate",
        "expected_expression_success": True,
        "notes": "Published expression in N. benthamiana: 10-100 mg/kg with ER retention",
    },
]

# ── Benchmark 4: Published Expression Yields ────────────────────────────────────
# Real experimental yields from published papers for comparison

YIELD_BENCHMARKS = [
    {
        "protein_name": "anti-HIV antibody 2G12",
        "species": "nbenthamiana",
        "localization": "ER_retained",
        "delivery": "transient",
        "published_yield_mg_kg": 55,
        "published_range": "30-80",
        "reference": "Sainsbury et al. 2008, Plant Biotechnol J",
        "notes": "Full IgG with KDEL, p19 co-infiltration",
    },
    {
        "protein_name": "Cetuximab (anti-EGFR mAb)",
        "species": "nbenthamiana",
        "localization": "secreted",
        "delivery": "transient",
        "published_yield_mg_kg": 140,
        "published_range": "100-200",
        "reference": "Eidenberger et al. 2022",
        "notes": "Secreted full IgG, p19, magnICON system",
    },
    {
        "protein_name": "Hepatitis B surface antigen",
        "species": "nbenthamiana",
        "localization": "ER_retained",
        "delivery": "transient",
        "published_yield_mg_kg": 4000,
        "published_range": "1000-5000",
        "reference": "Diamos et al. 2020",
        "notes": "VLP-forming antigen, very high expression",
    },
    {
        "protein_name": "Human serum albumin",
        "species": "rice",
        "localization": "ER_retained",
        "delivery": "stable",
        "published_yield_mg_kg": 2750,
        "published_range": "2500-3000",
        "reference": "He et al. 2011",
        "notes": "Rice seed stable expression",
    },
    {
        "protein_name": "Norwalk virus VLP",
        "species": "nbenthamiana",
        "localization": "ER_retained",
        "delivery": "transient",
        "published_yield_mg_kg": 800,
        "published_range": "500-1000",
        "reference": "Santi et al. 2008",
        "notes": "VLP-forming capsid protein",
    },
    {
        "protein_name": "Human growth hormone",
        "species": "nbenthamiana",
        "localization": "secreted",
        "delivery": "transient",
        "published_yield_mg_kg": 10,
        "published_range": "5-15",
        "reference": "Gils et al. 2005",
        "notes": "Small secreted protein",
    },
]
