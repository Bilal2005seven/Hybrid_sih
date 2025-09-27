"""
Crop Recommendation Services
Complete hybrid system with ML + LangChain + Root Agent
"""

from .pipeline import (
    CropRecommendationPipeline,
    recommend_crops,
    get_hybrid_recommendations,
    get_simple_recommendations,
    pipeline
)

from .main_crop_system import (
    get_crop_recommendations as get_main_crop_recommendations,
    get_simple_crop_recommendations as get_main_simple_recommendations
)

from .crop_recommendation_system import (
    get_hybrid_analysis,
    get_simple_recommendations as get_system_simple_recommendations
)

from .root_agent import (
    get_final_crop_recommendations,
    complete_hybrid_crop_analysis
)

__all__ = [
    # Main pipeline functions
    "CropRecommendationPipeline",
    "recommend_crops",
    "get_hybrid_recommendations", 
    "get_simple_recommendations",
    "pipeline",
    
    # Main crop system functions
    "get_main_crop_recommendations",
    "get_main_simple_recommendations",
    
    # Individual system functions
    "get_hybrid_analysis",
    "get_system_simple_recommendations",
    
    # Root agent functions
    "get_final_crop_recommendations",
    "complete_hybrid_crop_analysis"
]
