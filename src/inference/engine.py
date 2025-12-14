from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from shapely.geometry import shape, Point, Polygon
from shapely.ops import unary_union
import json

from ..scoring.aggregator import ScoreAggregator
from ..scoring.heuristics import (
    ElevationHeuristic,
    SnowConditionsHeuristic,
    WaterDistanceHeuristic,
    VegetationQualityHeuristic,
    HuntingPressureHeuristic,
    SecurityHabitatHeuristic,
    PredationRiskHeuristic,
    NutritionalConditionHeuristic,
    WinterSeverityHeuristic
)
from ..data.processors import DataContextBuilder

class ElkPredictionEngine:
    """Main inference engine for elk predictions"""
    
    def __init__(self, data_dir: str, weights: Optional[Dict] = None, 
                 method: str = "additive"):
        """
        Initialize prediction engine
        
        Args:
            data_dir: Path to data directory
            weights: Optional heuristic weights
            method: "additive" or "multiplicative"
        """
        self.data_dir = Path(data_dir)
        self.data_builder = DataContextBuilder(self.data_dir)
        
        # Set up heuristics with weights
        default_weights = weights or {}
        self.heuristics = [
            ElevationHeuristic(weight=default_weights.get("elevation", 1.0)),
            SnowConditionsHeuristic(weight=default_weights.get("snow", 1.0)),
            WaterDistanceHeuristic(weight=default_weights.get("water", 1.0)),
            VegetationQualityHeuristic(weight=default_weights.get("vegetation", 1.0)),
            HuntingPressureHeuristic(weight=default_weights.get("hunting_pressure", 1.0)),
            SecurityHabitatHeuristic(weight=default_weights.get("security", 1.0)),
            PredationRiskHeuristic(weight=default_weights.get("predation", 1.0)),
            NutritionalConditionHeuristic(weight=default_weights.get("nutrition", 1.0)),
            WinterSeverityHeuristic(weight=default_weights.get("winterkill", 1.0))
        ]
        
        # Create aggregator
        self.aggregator = ScoreAggregator(self.heuristics, method=method)
        
        # Cache for expensive operations
        self._grid_cache = {}
    
    def predict(self, request: Dict) -> Dict:
        """
        Main prediction entry point
        
        Args:
            request: {
                "location": {"type": "polygon", "coordinates": [...]},
                "date_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD", 
                              "find_best_days": bool},
                "radius_miles": float (optional),
                ...
            }
        
        Returns:
            Complete prediction response
        """
        print(f"\n{'='*60}")
        print(f"PathWild Elk Prediction")
        print(f"{'='*60}\n")
        
        # Parse request
        polygon = self._parse_polygon(request["location"]["coordinates"])
        date_range = request["date_range"]
        dates = self._generate_date_range(date_range["start"], date_range["end"])
        
        print(f"Area: {self._calculate_area_acres(polygon):.0f} acres")
        print(f"Dates: {date_range['start']} to {date_range['end']} ({len(dates)} days)")
        print(f"Method: {self.aggregator.method}")
        print()
        
        # Calculate spatial scores for each date
        print("Calculating spatial predictions...")
        spatial_scores = {}
        all_point_results = {}
        
        for i, date in enumerate(dates):
            print(f"  [{i+1}/{len(dates)}] {date}...", end=" ")
            scores, point_results = self._calculate_polygon_scores(polygon, date)
            spatial_scores[date] = scores
            all_point_results[date] = point_results
            print(f"avg score: {np.mean(scores):.1f}")
        
        print()
        
        # Build response
        response = {
            "query": {
                "area_acres": self._calculate_area_acres(polygon),
                "area_sq_miles": self._calculate_area_acres(polygon) / 640,
                "center": self._get_polygon_center(polygon),
                "date_range": date_range,
                "days_analyzed": len(dates)
            },
            "overall": self._calculate_overall_metrics(spatial_scores, polygon),
            "hotspots": self._identify_hotspots(spatial_scores, all_point_results, 
                                                polygon, dates),
            "factor_breakdown": self._aggregate_factor_breakdown(all_point_results, dates),
            "migration_status": self._assess_migration_status(dates[0]),
            "activity_patterns": self._generate_activity_patterns(),
            "recommendations": self._generate_recommendations(spatial_scores, polygon, dates),
            "metadata": self._build_metadata()
        }
        
        # Add daily rankings if requested
        if date_range.get("find_best_days", False):
            print("Ranking best days...")
            response["daily_predictions"] = self._rank_days(
                spatial_scores, all_point_results, dates, polygon
            )
        
        print(f"\n{'='*60}")
        print(f"Prediction complete!")
        print(f"Overall score: {response['overall']['score']:.1f}/100")
        print(f"Estimated population: {response['overall']['estimated_population']} elk")
        print(f"{'='*60}\n")
        
        return response
    
    def _parse_polygon(self, coordinates: List) -> Polygon:
        """Parse GeoJSON coordinates to Shapely Polygon"""
        return shape({"type": "Polygon", "coordinates": coordinates})
    
    def _generate_date_range(self, start: str, end: str) -> List[str]:
        """Generate list of dates between start and end"""
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        
        dates = []
        current = start_dt
        while current <= end_dt:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        return dates
    
    def _calculate_polygon_scores(self, polygon: Polygon, date: str) \
            -> Tuple[np.ndarray, List[Dict]]:
        """
        Calculate scores for all points within polygon
        
        Returns:
            scores: Array of aggregate scores
            point_results: List of full results for each point
        """
        # Generate sampling grid
        points = self._generate_grid_points(polygon, spacing_meters=500)
        
        scores = []
        point_results = []
        
        for point in points:
            location = {"lat": point.y, "lon": point.x}
            
            # Build context
            context = self.data_builder.build_context(location, date)
            
            # Calculate aggregate score
            aggregate = self.aggregator.calculate_aggregate(location, date, context)
            
            scores.append(aggregate.total_score)
            point_results.append({
                "location": location,
                "score": aggregate.total_score,
                "confidence": aggregate.confidence,
                "factor_scores": aggregate.factor_scores,
                "limiting_factor": aggregate.limiting_factor,
                "best_feature": aggregate.best_feature
            })
        
        return np.array(scores), point_results
    
    def _generate_grid_points(self, polygon: Polygon, spacing_meters: int = 500) \
            -> List[Point]:
        """Generate regular grid of points within polygon"""
        
        # Get bounding box
        minx, miny, maxx, maxy = polygon.bounds
        
        # Convert spacing to degrees (approximate)
        spacing_deg = spacing_meters / 111139.0  # meters to degrees
        
        # Generate grid
        points = []
        x = minx
        while x <= maxx:
            y = miny
            while y <= maxy:
                point = Point(x, y)
                if polygon.contains(point):
                    points.append(point)
                y += spacing_deg
            x += spacing_deg
        
        return points
    
    def _calculate_area_acres(self, polygon: Polygon) -> float:
        """Calculate polygon area in acres"""
        # Convert from square degrees to square meters to acres
        # This is approximate - for production use proper projection
        area_sq_deg = polygon.area
        area_sq_m = area_sq_deg * (111139 ** 2)  # degrees to meters
        area_acres = area_sq_m / 4046.86  # sq meters to acres
        return area_acres
    
    def _get_polygon_center(self, polygon: Polygon) -> Dict:
        """Get center point of polygon"""
        centroid = polygon.centroid
        return {"lat": centroid.y, "lon": centroid.x}
    
    def _calculate_overall_metrics(self, spatial_scores: Dict, polygon: Polygon) -> Dict:
        """Calculate aggregate metrics across all dates"""
        
        all_scores = np.concatenate(list(spatial_scores.values()))
        avg_score = float(np.mean(all_scores))
        
        # Estimate population based on score and area
        area_sq_miles = self._calculate_area_acres(polygon) / 640
        
        # Population density model: higher scores = more elk
        # Typical: 2-10 elk per square mile
        base_density = 2.0 + (avg_score / 100) * 8.0  # 2-10 range
        estimated_pop = int(base_density * area_sq_miles)
        
        # Confidence level
        if avg_score >= 75:
            confidence = "high"
        elif avg_score >= 60:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Habitat quality
        if avg_score >= 80:
            quality = "excellent"
        elif avg_score >= 65:
            quality = "good"
        elif avg_score >= 50:
            quality = "fair"
        else:
            quality = "poor"
        
        return {
            "score": avg_score,
            "confidence_level": confidence,
            "estimated_population": estimated_pop,
            "population_density": base_density,
            "habitat_quality": quality
        }
    
    def _identify_hotspots(self, spatial_scores: Dict, all_point_results: Dict,
                          polygon: Polygon, dates: List[str]) -> List[Dict]:
        """Identify top elk concentration areas"""
        
        print("Identifying hotspots...")
        
        # Combine all point results across dates
        all_points = []
        for date in dates:
            for result in all_point_results[date]:
                all_points.append({
                    **result,
                    "date": date
                })
        
        # Sort by score
        all_points.sort(key=lambda x: x["score"], reverse=True)
        
        # Cluster nearby high-scoring points
        hotspots = []
        used_points = set()
        
        for point in all_points:
            if len(hotspots) >= 10:
                break
            
            point_id = f"{point['location']['lat']:.4f},{point['location']['lon']:.4f}"
            if point_id in used_points:
                continue
            
            # Find nearby points (cluster)
            cluster = self._find_nearby_points(point, all_points, radius_deg=0.01)
            
            # Mark as used
            for p in cluster:
                pid = f"{p['location']['lat']:.4f},{p['location']['lon']:.4f}"
                used_points.add(pid)
            
            # Calculate cluster metrics
            cluster_scores = [p["score"] for p in cluster]
            avg_cluster_score = np.mean(cluster_scores)
            
            # Estimate elk in cluster
            cluster_area_acres = len(cluster) * 0.25  # Rough estimate
            elk_density = 2.0 + (avg_cluster_score / 100) * 8.0
            estimated_elk_min = int(elk_density * 0.8 * (cluster_area_acres / 640))
            estimated_elk_max = int(elk_density * 1.2 * (cluster_area_acres / 640))
            estimated_elk_mean = int(elk_density * (cluster_area_acres / 640))
            
            # Ensure min is at least 1, and max is at least min
            min_elk = max(1, estimated_elk_min)
            max_elk = max(min_elk, estimated_elk_max)  # Ensure max >= min
            
            hotspot = {
                "id": f"hotspot_{len(hotspots) + 1}",
                "rank": len(hotspots) + 1,
                "center": point["location"],
                "area_acres": cluster_area_acres,
                "score": avg_cluster_score,
                "estimated_elk": {
                    "min": min_elk,
                    "max": max_elk,
                    "mean": max(1, estimated_elk_mean)
                },
                "description": self._generate_hotspot_description(point),
                "limiting_factor": point["limiting_factor"],
                "best_feature": point["best_feature"]
            }
            
            hotspots.append(hotspot)
        
        print(f"  Found {len(hotspots)} hotspots")
        
        return hotspots
    
    def _find_nearby_points(self, center_point: Dict, all_points: List[Dict], 
                           radius_deg: float = 0.01) -> List[Dict]:
        """Find all points within radius of center"""
        center_lat = center_point["location"]["lat"]
        center_lon = center_point["location"]["lon"]
        
        nearby = []
        for point in all_points:
            lat = point["location"]["lat"]
            lon = point["location"]["lon"]
            
            dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            if dist <= radius_deg:
                nearby.append(point)
        
        return nearby
    
    def _generate_hotspot_description(self, point: Dict) -> str:
        """Generate text description of hotspot"""
        # Use factor scores to describe
        factors = point["factor_scores"]
        
        descriptions = []
        
        # Elevation
        if "elevation" in factors:
            elev = factors["elevation"].get("raw_value", 8500)
            descriptions.append(f"{elev:.0f}ft elevation")
        
        # Terrain
        if factors.get("security_habitat", {}).get("score", 0) > 7:
            descriptions.append("secure terrain")
        
        # Vegetation
        if factors.get("vegetation_quality", {}).get("score", 0) > 7:
            descriptions.append("good forage")
        
        return ", ".join(descriptions) if descriptions else "elk habitat"
    
    def _aggregate_factor_breakdown(self, all_point_results: Dict, 
                                    dates: List[str]) -> Dict:
        """Aggregate factor scores across all points and dates"""
        
        # Collect all factor scores
        factor_aggregates = {}
        
        for date in dates:
            for result in all_point_results[date]:
                for factor_name, factor_data in result["factor_scores"].items():
                    if factor_name not in factor_aggregates:
                        factor_aggregates[factor_name] = {
                            "scores": [],
                            "contributions": [],
                            "statuses": []
                        }
                    
                    factor_aggregates[factor_name]["scores"].append(
                        factor_data["score"]
                    )
                    factor_aggregates[factor_name]["contributions"].append(
                        factor_data["contribution"]
                    )
                    factor_aggregates[factor_name]["statuses"].append(
                        factor_data["status"]
                    )
        
        # Calculate averages
        breakdown = {}
        for factor_name, data in factor_aggregates.items():
            avg_score = float(np.mean(data["scores"]))
            avg_contribution = float(np.mean(data["contributions"]))
            
            # Most common status
            statuses = data["statuses"]
            most_common_status = max(set(statuses), key=statuses.count)
            
            breakdown[factor_name] = {
                "score": avg_score,
                "weight": self._get_heuristic_weight(factor_name),
                "contribution": avg_contribution,
                "status": most_common_status,
                "note": self._generate_factor_note(factor_name, avg_score, 
                                                   most_common_status)
            }
        
        return breakdown
    
    def _get_heuristic_weight(self, name: str) -> float:
        """Get weight for a heuristic by name"""
        for h in self.heuristics:
            if h.name == name:
                return h.weight
        return 1.0
    
    def _generate_factor_note(self, factor_name: str, score: float, 
                             status: str) -> str:
        """Generate explanatory note for factor"""
        if status == "excellent":
            return f"{factor_name.replace('_', ' ').title()} is optimal"
        elif status == "good":
            return f"{factor_name.replace('_', ' ').title()} is favorable"
        elif status == "fair":
            return f"{factor_name.replace('_', ' ').title()} is acceptable"
        else:
            return f"{factor_name.replace('_', ' ').title()} is limiting"
    
    def _assess_migration_status(self, date: str) -> Dict:
        """Assess migration status for date"""
        dt = datetime.fromisoformat(date)
        month = dt.month
        
        # Simple model based on month
        if month in [6, 7, 8]:
            phase = "summer_range"
            trend = "stable"
            days_until_migration = (datetime(dt.year, 10, 15) - dt).days
        elif month in [9, 10]:
            phase = "pre_migration"
            trend = "stable"
            days_until_migration = max(0, (datetime(dt.year, 11, 1) - dt).days)
        elif month == 11:
            phase = "active_migration"
            trend = "decreasing"
            days_until_migration = 0
        elif month in [12, 1, 2, 3]:
            phase = "winter_range"
            trend = "stable"
            days_until_migration = (datetime(dt.year if month > 3 else dt.year + 1, 4, 1) - dt).days
        else:  # 4, 5
            phase = "spring_migration"
            trend = "increasing"
            days_until_migration = 0
        
        return {
            "current_phase": phase,
            "expected_migration_date": "2026-11-01" if phase == "pre_migration" else None,
            "days_until_migration": days_until_migration if days_until_migration > 0 else None,
            "confidence": "medium",
            "trend": trend
        }
    
    def _generate_activity_patterns(self) -> Dict:
        """Generate time-of-day activity recommendations"""
        return {
            "dawn": {
                "score": 92,
                "description": "Peak activity - elk feeding in meadows and edges"
            },
            "morning": {
                "score": 78,
                "description": "Elk moving to bedding areas in timber"
            },
            "midday": {
                "score": 45,
                "description": "Elk bedded in security cover"
            },
            "afternoon": {
                "score": 65,
                "description": "Elk beginning to stir, stretching"
            },
            "dusk": {
                "score": 88,
                "description": "Evening feeding activity increasing"
            },
            "recommended_times": ["dawn", "dusk"]
        }
    
    def _generate_recommendations(self, spatial_scores: Dict, polygon: Polygon,
                                 dates: List[str]) -> List[Dict]:
        """Generate actionable hunting recommendations"""
        
        recommendations = []
        
        # Best areas (from hotspots that would be calculated)
        recommendations.append({
            "priority": "high",
            "category": "location",
            "text": "Focus on north-facing drainages with aspen/meadow mix"
        })
        
        # Best times
        recommendations.append({
            "priority": "high",
            "category": "timing",
            "text": "Hunt during dawn and dusk for highest elk activity"
        })
        
        # Strategy
        recommendations.append({
            "priority": "medium",
            "category": "strategy",
            "text": "Glass open meadows at dawn, still-hunt timber mid-morning"
        })
        
        return recommendations
    
    def _rank_days(self, spatial_scores: Dict, all_point_results: Dict,
                  dates: List[str], polygon: Polygon) -> List[Dict]:
        """Rank dates by predicted elk activity/presence"""
        
        daily_predictions = []
        
        for date in dates:
            scores = spatial_scores[date]
            avg_score = float(np.mean(scores))
            
            # Estimate population for this day
            area_sq_miles = self._calculate_area_acres(polygon) / 640
            density = 2.0 + (avg_score / 100) * 8.0
            population = int(density * area_sq_miles)
            
            # Generate reason
            reason = self._explain_daily_score(date, avg_score, all_point_results[date])
            
            # Confidence based on forecast horizon
            dt = datetime.fromisoformat(date)
            today = datetime.now().date()
            days_out = (dt.date() - today).days
            
            if days_out <= 3:
                confidence = "high"
            elif days_out <= 7:
                confidence = "medium"
            else:
                confidence = "low"
            
            daily_predictions.append({
                "date": date,
                "score": avg_score,
                "estimated_population": population,
                "reason": reason,
                "confidence": confidence
            })
        
        # Sort by score
        daily_predictions.sort(key=lambda x: x["score"], reverse=True)
        
        # Add ranks
        for i, pred in enumerate(daily_predictions):
            pred["rank"] = i + 1
        
        return daily_predictions
    
    def _explain_daily_score(self, date: str, score: float, 
                            point_results: List[Dict]) -> str:
        """Generate explanation for why this day scored well/poorly"""
        
        # Find dominant factors
        factor_scores = {}
        for result in point_results:
            for factor_name, factor_data in result["factor_scores"].items():
                if factor_name not in factor_scores:
                    factor_scores[factor_name] = []
                factor_scores[factor_name].append(factor_data["score"])
        
        # Find best and worst factors
        avg_factors = {k: np.mean(v) for k, v in factor_scores.items()}
        best_factor = max(avg_factors.items(), key=lambda x: x[1])
        worst_factor = min(avg_factors.items(), key=lambda x: x[1])
        
        if score >= 80:
            return f"Excellent conditions - {best_factor[0].replace('_', ' ')} is optimal"
        elif score >= 65:
            return f"Good conditions overall"
        elif score >= 50:
            return f"Fair conditions - {worst_factor[0].replace('_', ' ')} is limiting"
        else:
            return f"Poor conditions - {worst_factor[0].replace('_', ' ')} is suboptimal"
    
    def _build_metadata(self) -> Dict:
        """Build metadata about prediction quality"""
        
        return {
            "prediction_id": f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "model_version": "v0.1.0",
            "overall_confidence": 0.75,
            "confidence_factors": {
                "data_recency": 0.80,
                "forecast_horizon": 0.75,
                "spatial_coverage": 0.85,
                "historical_validation": 0.60
            },
            "data_sources": [
                "USGS DEM (terrain)",
                "NLCD (land cover)",
                "SNOTEL (snow)",
                "NOAA (weather)",
                "Landsat (NDVI)",
                "Wyoming Game & Fish (wildlife)"
            ],
            "limitations": [
                "Weather forecasts >7 days have reduced accuracy",
                "Population estimates based on habitat quality models",
                "Real-time elk locations not available"
            ]
        }