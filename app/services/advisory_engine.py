"""
NagarMitra - Phase 1
Health Advisory Engine
Generates personalized health advisories based on AQI level.
Bilingual: English + Hindi
"""

from typing import Optional


# ─── Advisory Data ────────────────────────────────────────────────────────────

ADVISORIES = {
    "Good": {
        "en": {
            "general":    "Air quality is good. Enjoy outdoor activities.",
            "sensitive":  "Safe for all, including sensitive groups.",
            "children":   "Children can play outdoors freely.",
            "elderly":    "Enjoy outdoor walks and activities.",
            "mask":       "No mask required.",
            "activities": ["Outdoor exercise", "Walk in park", "Open windows for fresh air"],
            "color_alert": "green",
        },
        "hi": {
            "general":    "वायु गुणवत्ता अच्छी है। बाहरी गतिविधियों का आनंद लें।",
            "sensitive":  "सभी के लिए सुरक्षित, संवेदनशील लोगों सहित।",
            "children":   "बच्चे स्वतंत्र रूप से बाहर खेल सकते हैं।",
            "elderly":    "बुजुर्ग बाहर टहल सकते हैं।",
            "mask":       "मास्क की जरूरत नहीं।",
            "activities": ["बाहर व्यायाम", "पार्क में टहलना", "ताजी हवा के लिए खिड़कियां खोलें"],
            "color_alert": "हरा",
        }
    },
    "Satisfactory": {
        "en": {
            "general":    "Air quality is acceptable but may affect very sensitive individuals.",
            "sensitive":  "Asthma/heart patients: limit prolonged outdoor exertion.",
            "children":   "Outdoor play is fine. Watch for symptoms.",
            "elderly":    "Short walks are okay. Avoid strenuous activity.",
            "mask":       "Optional for sensitive groups.",
            "activities": ["Light outdoor exercise", "Avoid jogging near busy roads"],
            "color_alert": "light_green",
        },
        "hi": {
            "general":    "वायु गुणवत्ता स्वीकार्य है, पर अति संवेदनशील लोगों को सावधान रहना चाहिए।",
            "sensitive":  "अस्थमा/हृदय रोगी: लंबे समय तक बाहर न रहें।",
            "children":   "बाहर खेल सकते हैं। लक्षणों पर ध्यान दें।",
            "elderly":    "छोटी सैर ठीक है। कठिन गतिविधि से बचें।",
            "mask":       "संवेदनशील लोगों के लिए वैकल्पिक।",
            "activities": ["हल्का बाहरी व्यायाम", "व्यस्त सड़कों के पास जॉगिंग से बचें"],
            "color_alert": "हल्का हरा",
        }
    },
    "Moderate": {
        "en": {
            "general":    "People with lung/heart conditions may experience breathing discomfort.",
            "sensitive":  "Avoid outdoor activities. Keep inhalers handy.",
            "children":   "Limit outdoor playtime. Prefer indoor activities.",
            "elderly":    "Stay indoors if possible. Consult doctor if breathless.",
            "mask":       "N95 mask recommended for sensitive groups.",
            "activities": ["Indoor exercises preferred", "Keep windows partially closed", "Use air purifier if available"],
            "color_alert": "yellow",
        },
        "hi": {
            "general":    "फेफड़े/हृदय रोगियों को सांस लेने में तकलीफ हो सकती है।",
            "sensitive":  "बाहरी गतिविधियों से बचें। इन्हेलर पास रखें।",
            "children":   "बाहर खेलने का समय सीमित करें। इनडोर गतिविधियां पसंद करें।",
            "elderly":    "यदि संभव हो तो घर के अंदर रहें। सांस की तकलीफ होने पर डॉक्टर से मिलें।",
            "mask":       "संवेदनशील लोगों के लिए N95 मास्क अनुशंसित।",
            "activities": ["इनडोर व्यायाम बेहतर है", "खिड़कियां आंशिक रूप से बंद रखें", "एयर प्यूरीफायर उपयोग करें"],
            "color_alert": "पीला",
        }
    },
    "Poor": {
        "en": {
            "general":    "Everyone may begin to experience adverse health effects. Limit outdoor exposure.",
            "sensitive":  "Stay indoors. Avoid all outdoor exertion.",
            "children":   "Avoid outdoor activities. School should consider indoor recess.",
            "elderly":    "Stay indoors. Keep doors and windows closed.",
            "mask":       "N95 mask mandatory if going outside.",
            "activities": ["Avoid all outdoor exercise", "Keep windows closed", "Run air purifier", "Work from home if possible"],
            "color_alert": "orange",
        },
        "hi": {
            "general":    "सभी को स्वास्थ्य पर प्रतिकूल प्रभाव हो सकता है। बाहर कम निकलें।",
            "sensitive":  "घर के अंदर रहें। सभी बाहरी गतिविधियों से बचें।",
            "children":   "बाहरी गतिविधियों से बचें। स्कूल में इनडोर खेल पसंद करें।",
            "elderly":    "घर के अंदर रहें। दरवाजे और खिड़कियां बंद रखें।",
            "mask":       "बाहर जाने पर N95 मास्क अनिवार्य।",
            "activities": ["सभी बाहरी व्यायाम से बचें", "खिड़कियां बंद रखें", "एयर प्यूरीफायर चलाएं", "हो सके तो घर से काम करें"],
            "color_alert": "नारंगी",
        }
    },
    "Very Poor": {
        "en": {
            "general":    "Serious health risk. Avoid all outdoor activity. Authorities should take action.",
            "sensitive":  "Seek medical attention if experiencing symptoms. Do not go outside.",
            "children":   "Schools may consider cancelling outdoor activities. Keep children indoors.",
            "elderly":    "Do not go outside at all. Keep medicines ready.",
            "mask":       "N95/N99 mask mandatory. Avoid any outdoor exposure.",
            "activities": ["Do not go outside", "Seal gaps in windows/doors", "Run air purifier at highest setting", "Drink plenty of water"],
            "color_alert": "red",
        },
        "hi": {
            "general":    "गंभीर स्वास्थ्य खतरा। सभी बाहरी गतिविधियों से बचें।",
            "sensitive":  "लक्षण होने पर तुरंत डॉक्टर से मिलें। बाहर न जाएं।",
            "children":   "स्कूल बाहरी गतिविधियां रद्द कर सकते हैं। बच्चों को घर में रखें।",
            "elderly":    "बिल्कुल बाहर न जाएं। दवाइयां तैयार रखें।",
            "mask":       "N95/N99 मास्क अनिवार्य। बाहरी संपर्क से बिल्कुल बचें।",
            "activities": ["बाहर न जाएं", "खिड़की/दरवाजों की दरारें बंद करें", "एयर प्यूरीफायर पूरी शक्ति पर चलाएं", "खूब पानी पिएं"],
            "color_alert": "लाल",
        }
    },
    "Severe": {
        "en": {
            "general":    "EMERGENCY LEVEL. Affects healthy people. Severe respiratory risk. Stay indoors.",
            "sensitive":  "Emergency medical alert. Avoid any exposure. Call doctor immediately if breathless.",
            "children":   "Do not send children outside. School should be closed or shifted online.",
            "elderly":    "Health emergency. Remain indoors. Keep emergency contacts ready.",
            "mask":       "N99/P100 respirator needed. Minimize all outdoor time.",
            "activities": ["Stay indoors completely", "Emergency inhalers ready", "Emergency helpline: 14411 (AIIMS Delhi)", "Close all ventilation"],
            "color_alert": "maroon",
        },
        "hi": {
            "general":    "आपातकालीन स्तर। स्वस्थ लोगों को भी प्रभावित करता है। घर के अंदर रहें।",
            "sensitive":  "चिकित्सा आपातकाल। किसी भी संपर्क से बचें। सांस में तकलीफ होने पर तुरंत डॉक्टर बुलाएं।",
            "children":   "बच्चों को बाहर न भेजें। स्कूल बंद या ऑनलाइन हो जाएं।",
            "elderly":    "स्वास्थ्य आपातकाल। घर में रहें। आपातकालीन संपर्क तैयार रखें।",
            "mask":       "N99/P100 रेस्पिरेटर जरूरी। बाहरी समय कम से कम करें।",
            "activities": ["पूरी तरह घर के अंदर रहें", "आपातकालीन इन्हेलर तैयार रखें", "हेल्पलाइन: 14411 (AIIMS दिल्ली)", "सभी वेंटिलेशन बंद करें"],
            "color_alert": "मैरून",
        }
    },
}


def get_health_advisory(
    aqi: Optional[float],
    ward_name: str,
    profile: Optional[str] = "general",   # general | sensitive | children | elderly
    lang: str = "en",
) -> dict:
    """
    Get a personalized health advisory for a ward.

    Args:
        aqi: Current AQI value
        ward_name: Name of the Delhi ward
        profile: User profile type
        lang: 'en' for English, 'hi' for Hindi, 'both' for both

    Returns:
        Dict with advisory text, recommendations, and alert color
    """
    if aqi is None:
        return {"ward": ward_name, "error": "AQI data unavailable"}

    category = _aqi_to_category(aqi)
    advisory_set = ADVISORIES.get(category, ADVISORIES["Moderate"])

    def _build(language: str) -> dict:
        data = advisory_set.get(language, advisory_set["en"])
        prof_key = profile if profile in data else "general"
        return {
            "category":         category,
            "aqi":              aqi,
            "ward":             ward_name,
            "alert_color":      data.get("color_alert"),
            "message":          data.get(prof_key, data["general"]),
            "general_advice":   data.get("general"),
            "mask_advice":      data.get("mask"),
            "recommended_actions": data.get("activities", []),
            "language":         language,
        }

    if lang == "both":
        return {
            "ward":  ward_name,
            "aqi":   aqi,
            "category": category,
            "en": _build("en"),
            "hi": _build("hi"),
        }

    return _build(lang)


def get_bulk_advisories(ward_aqis: list[dict], lang: str = "both") -> list[dict]:
    """Get advisories for multiple wards at once."""
    return [
        get_health_advisory(w.get("aqi"), w.get("ward", "Unknown"), lang=lang)
        for w in ward_aqis
    ]


def _aqi_to_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
    return "Severe"