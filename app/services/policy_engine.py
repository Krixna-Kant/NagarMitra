"""
policy_engine.py
NagarMitra Phase 4 — Civic Policy Recommendation Engine
Generates actionable administrative policies based on the Machine Learning source attribution.
"""

def get_policies_for_source(source_category: str) -> list:
    """
    Returns an array of actionable policy recommendations based on the root cause classifier.
    """
    category = str(source_category).lower().strip()
    
    policies = {
        "traffic": [
            "Implement Odd-Even vehicle rationing scheme immediately.",
            "Reroute all non-essential heavy commercial vehicles (HCVs) away from ward borders.",
            "Deploy traffic police to clear identified bottleneck junctions preventing idling.",
            "Subsidize or mandate electric public transit for the next 48 hours."
        ],
        "dust_construction": [
            "Halt all non-essential excavation and building construction in the ward.",
            "Mandate continuous deployment of Anti-Smog Guns at all active sites.",
            "Deploy mechanized road sweepers and water sprinklers on major arterial roads.",
            "Impose strict fines on uncovered construction material storage (NGT guidelines)."
        ],
        "biomass_burning": [
            "Deploy rapid-response flying squads to penalize illegal open waste/leaf burning.",
            "Coordinate with adjacent municipalities to track agricultural stubble fires via satellite.",
            "Initiate public awareness campaigns on the legal penalties of biomass burning.",
            "Provide subsidized mechanical alternatives for agricultural clearing if applicable."
        ],
        "industrial": [
            "Issue temporary closure notices to non-compliant red-category industrial units.",
            "Mandate the use of PNG (Piped Natural Gas) instead of pet-coke or coal.",
            "Increase DPCC randomized emission inspections by 200%.",
            "Restrict generator (DG set) usage to essential services only."
        ],
        "weather_trapped": [
            "Declare Public Health Emergency: advise sensitive groups to remain indoors.",
            "Halt schools and non-essential outdoor public gatherings.",
            "Meteorological constraints prevent dispersion; focus entirely on minimizing new emissions.",
            "Activate emergency medical staging areas for asthma/respiratory spikes."
        ]
    }
    
    # "unclassified" or unknown falls back to general
    general = [
        "Increase general AQI monitoring frequency.",
        "Advise citizens to use N95 masks outdoors.",
        "Prepare local health infrastructure for respiratory cases."
    ]
    
    return policies.get(category, general)
