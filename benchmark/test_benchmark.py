"""
Test script to verify benchmark system functionality.
Runs a quick test with reduced parameters to ensure everything works.
"""

import os
import sys
import time

# Add benchmark directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_data_availability():
    """Test if data files are available."""
    data_paths = {
        "red_wine": "data/winequality-red.csv",
        "white_wine": "data/winequality-white.csv",
    }

    missing_files = []
    for name, path in data_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    if missing_files:
        print("❌ Missing data files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ All data files found")
        return True


def test_library_imports():
    """Test if required libraries can be imported."""
    libraries = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "sklearn.ensemble",
        "hyperopt": "hyperopt",
        "optuna": "optuna",
        "ray": "ray",
    }

    available_libs = []

    for lib_name, import_name in libraries.items():
        try:
            __import__(import_name)
            print(f"✅ {lib_name} available")
            available_libs.append(lib_name)
        except ImportError as e:
            print(f"❌ {lib_name} not available: {e}")

    return available_libs


def test_benchmark_modules():
    """Test individual benchmark modules."""
    benchmark_modules = {
        "hyperopt_benchmark": "run_hyperopt_benchmark",
        "optuna_benchmark": "run_optuna_benchmark",
        "sklearn_benchmark": "run_sklearn_benchmark",
    }

    available_modules = []

    for module_name, function_name in benchmark_modules.items():
        try:
            module = __import__(module_name)
            # Check if the function exists
            if hasattr(module, function_name):
                available_modules.append(module_name)
                print(f"✅ {module_name} benchmark module loaded successfully")
            else:
                print(f"❌ {module_name} missing function {function_name}")
        except ImportError as e:
            print(f"❌ {module_name} benchmark module failed to load: {e}")

    return available_modules


def run_quick_test():
    """Run a quick test with minimal parameters."""
    print("🚀 Running Quick Benchmark Test")
    print("=" * 50)

    # Test data availability
    if not test_data_availability():
        print("\n❌ Cannot proceed without data files")
        return False

    # Test library imports
    print("\n📚 Testing library imports...")
    available_libs = test_library_imports()

    # Test module availability
    print("\n🔧 Testing benchmark modules...")
    available_modules = test_benchmark_modules()

    if not available_modules:
        print("\n❌ No benchmark modules available")
        return False

    print(f"\n📊 Available modules: {', '.join(available_modules)}")

    # Try to run a quick sklearn benchmark (usually most reliable)
    if "sklearn_benchmark" in available_modules:
        print("\n🧪 Running quick sklearn test...")
        try:
            from sklearn_benchmark import run_sklearn_benchmark

            data_paths = {"red_wine": "data/winequality-red.csv"}

            # Run with minimal parameters
            start_time = time.time()
            results = run_sklearn_benchmark(
                data_paths, n_iter=5, random_seed=42
            )  # Only 5 iterations
            test_time = time.time() - start_time

            if results and "red_wine" in results:
                print(f"✅ Quick test completed in {test_time:.2f}s")
                print("✅ Benchmark system is working correctly!")

                # Show a sample result
                red_results = results["red_wine"]
                if "classification" in red_results:
                    cls_result = red_results["classification"]
                    if "RandomizedSearch" in cls_result:
                        score = cls_result["RandomizedSearch"].get("final_score", "N/A")
                        print(f"   Sample result - Classification score: {score}")

                return True
            else:
                print("❌ Quick test failed - no results returned")
                return False

        except Exception as e:
            print(f"❌ Quick test failed with error: {e}")
            import traceback

            traceback.print_exc()
            return False

    else:
        print("\n⚠️  sklearn_benchmark not available for quick test")
        print("✅ Module loading test passed - benchmark system should work")
        return True


def main():
    """Main test function."""
    print("Wine Quality Benchmark Test")
    print("Testing benchmark system functionality...")
    print()

    success = run_quick_test()

    print("\n" + "=" * 50)
    if success:
        print("✅ BENCHMARK SYSTEM READY")
        print("\nYou can now run:")
        print("  uv run benchmark/benchmark_no_ft.py")
        print("\nOr individual benchmarks:")
        print("  uv run benchmark/hyperopt_benchmark.py")
        print("  uv run benchmark/optuna_benchmark.py")
        print("  uv run benchmark/sklearn_benchmark.py")
    else:
        print("❌ BENCHMARK SYSTEM NOT READY")
        print("\nPlease check:")
        print("  1. Install required dependencies: uv pip install -r requirements.txt")
        print("  2. Ensure data files are in data/ directory")
        print("  3. Check error messages above")


if __name__ == "__main__":
    main()
