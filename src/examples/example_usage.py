"""
Example usage of PathWild elk prediction system
"""

from pathlib import Path
from src.inference.engine import ElkPredictionEngine
import json

def main():
    # Initialize engine
    data_dir = Path("data")
    engine = ElkPredictionEngine(
        data_dir=str(data_dir),
        weights={
            "elevation": 2.5,
            "water": 3.0,
            "vegetation": 2.0,
            "hunting_pressure": 1.5,
            "security": 1.5,
            "snow": 2.0,
            "predation": 1.0,
            "nutrition": 1.0,
            "winterkill": 1.0
        }
    )
    
    # Create prediction request
    # Example: Hunt Area 87 in Wyoming (approximate coordinates)
    request = {
        "location": {
            "type": "polygon",
            "coordinates": [[
                [-110.75, 43.10],
                [-110.65, 43.10],
                [-110.65, 43.00],
                [-110.75, 43.00],
                [-110.75, 43.10]
            ]]
        },
        "date_range": {
            "start": "2026-10-27",
            "end": "2026-10-31",
            "find_best_days": True
        }
    }
    
    # Run prediction
    print("Running PathWild prediction...")
    response = engine.predict(request)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\nOverall Score: {response['overall']['score']:.1f}/100")
    print(f"Confidence: {response['overall']['confidence_level']}")
    print(f"Estimated Population: {response['overall']['estimated_population']} elk")
    print(f"Density: {response['overall']['population_density']:.1f} elk/sq mi")
    print(f"Habitat Quality: {response['overall']['habitat_quality']}")
    
    # Best days
    if "daily_predictions" in response:
        print("\nBEST HUNTING DAYS:")
        for pred in response["daily_predictions"][:3]:  # Top 3
            print(f"  #{pred['rank']}: {pred['date']} "
                  f"(score: {pred['score']:.1f}) - {pred['reason']}")
    
    # Top hotspots
    print("\nTOP HOTSPOTS:")
    for hotspot in response["hotspots"][:5]:  # Top 5
        print(f"  #{hotspot['rank']}: {hotspot['description']}")
        print(f"    Location: {hotspot['center']['lat']:.4f}, "
              f"{hotspot['center']['lon']:.4f}")
        print(f"    Score: {hotspot['score']:.1f}")
        print(f"    Est. Elk: {hotspot['estimated_elk']['min']}-"
              f"{hotspot['estimated_elk']['max']}")
    
    # Factor breakdown
    print("\nFACTOR BREAKDOWN:")
    for name, data in response["factor_breakdown"].items():
        print(f"  {name.replace('_', ' ').title()}: "
              f"{data['score']:.1f}/10 ({data['status']}) - {data['note']}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    for rec in response["recommendations"]:
        print(f"  [{rec['priority'].upper()}] {rec['text']}")
    
    # Save to file
    output_path = Path("output") / "prediction_results.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(response, f, indent=2)
    
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()