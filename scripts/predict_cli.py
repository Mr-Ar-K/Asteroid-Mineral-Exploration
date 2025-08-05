#!/usr/bin/env python3
"""
Command-line interface for asteroid mining predictions.
"""
import sys
from pathlib import Path
import argparse

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.predict import AsteroidPredictor

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="AI-Driven Asteroid Mining Potential Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_cli.py --asteroid "2000 SG344"
  python predict_cli.py --asteroid "433 Eros" --report
  python predict_cli.py --list "2000 SG344,2019 GT3,2020 BX12"
        """
    )
    
    parser.add_argument(
        "--asteroid", "-a",
        type=str,
        help="Single asteroid designation to analyze"
    )
    
    parser.add_argument(
        "--list", "-l",
        type=str,
        help="Comma-separated list of asteroid designations"
    )
    
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate detailed assessment report"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results"
    )
    
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=10,
        help="Number of top asteroids to show (for list mode)"
    )
    
    args = parser.parse_args()
    
    if not args.asteroid and not args.list:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize predictor
        print("🤖 Initializing AI models...")
        predictor = AsteroidPredictor()
        print("✅ Models loaded successfully")
        
        if args.asteroid:
            # Single asteroid analysis
            print(f"\n🔍 Analyzing asteroid: {args.asteroid}")
            
            if args.report:
                report = predictor.generate_mining_report(args.asteroid, args.output)
                print(report)
            else:
                result = predictor.predict_single_asteroid(args.asteroid)
                if result:
                    display_single_result(result)
                else:
                    print(f"❌ Analysis failed for {args.asteroid}")
        
        elif args.list:
            # Multiple asteroid analysis
            asteroids = [a.strip() for a in args.list.split(',')]
            print(f"\n🔍 Analyzing {len(asteroids)} asteroids...")
            
            top_asteroids = predictor.rank_asteroids_by_mining_potential(
                asteroids, top_n=args.top
            )
            
            if top_asteroids:
                display_ranking_results(top_asteroids)
                
                if args.output:
                    save_results_to_file(top_asteroids, args.output)
            else:
                print("❌ No successful analyses")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def display_single_result(result):
    """Display results for a single asteroid."""
    designation = result['designation']
    mining_score = result['mining_assessment']['mining_potential_score']
    category = result['mining_assessment']['mining_category']
    confidence = result['mining_assessment']['confidence_score']
    
    print(f"\n{'='*50}")
    print(f"🌌 ASTEROID: {designation}")
    print(f"{'='*50}")
    print(f"🎯 Mining Potential: {category.upper()} ({mining_score:.3f}/1.0)")
    print(f"🔮 Confidence: {confidence:.1%}")
    print(f"📏 Diameter: {result['basic_info']['diameter_km']:.3f} km")
    print(f"🛰️ Delta-V: {result['derived_metrics']['delta_v_total_km_s']:.1f} km/s")
    print(f"🚀 Accessibility: {result['derived_metrics']['accessibility_score']:.3f}")
    print(f"💰 Economic Value: {result['derived_metrics']['economic_value']:.3f}")
    
    # Mining potential indicator
    if category in ['high', 'very_high']:
        print("🟢 RECOMMENDED for mining consideration")
    elif category == 'medium':
        print("🟡 MODERATE potential - requires detailed analysis")
    else:
        print("🔴 LOW potential - not recommended")

def display_ranking_results(top_asteroids):
    """Display ranking results for multiple asteroids."""
    print(f"\n{'='*80}")
    print(f"🏆 TOP {len(top_asteroids)} ASTEROID MINING TARGETS")
    print(f"{'='*80}")
    
    print(f"{'Rank':<4} {'Asteroid':<12} {'Score':<6} {'Category':<8} {'Delta-V':<8} {'Diameter':<10}")
    print("-" * 80)
    
    for asteroid in top_asteroids:
        rank = asteroid['rank']
        designation = asteroid['designation'][:11]  # Truncate for display
        score = asteroid['mining_assessment']['mining_potential_score']
        category = asteroid['mining_assessment']['mining_category'][:7]
        delta_v = asteroid['derived_metrics']['delta_v_total_km_s']
        diameter = asteroid['basic_info']['diameter_km']
        
        print(f"{rank:<4} {designation:<12} {score:<6.3f} {category:<8} {delta_v:<8.1f} {diameter:<10.3f}")
    
    print("-" * 80)
    print(f"Analysis complete! Top candidate: {top_asteroids[0]['designation']}")

def save_results_to_file(results, filename):
    """Save results to file."""
    import json
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()
