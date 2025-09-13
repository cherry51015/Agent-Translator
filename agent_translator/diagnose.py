#!/usr/bin/env python3
"""
CrowdWisdomTrading Translation System Diagnostic Script
Helps identify and fix common setup issues
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 10):
        print("  ✅ Python version is compatible")
        return True
    else:
        print("  ❌ Python 3.10+ required")
        return False

def check_project_structure():
    """Verify project directory structure"""
    print("Checking project structure...")
    
    required_files = {
        "config.yaml": "Configuration file",
        "src/agent_translator/__init__.py": "Main package init",
        "src/agent_translator/crew.py": "Crew implementation",
        "src/agent_translator/tools/__init__.py": "Tools package init", 
        "src/agent_translator/tools/custom_tools.py": "Custom tools implementation",
        "main.py": "CLI interface"
    }
    
    required_dirs = [
        "src/agent_translator",
        "src/agent_translator/tools",
        "output"
    ]
    
    structure_ok = True
    
    # Check directories
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"  ✅ Directory: {directory}")
        else:
            print(f"  ❌ Missing directory: {directory}")
            structure_ok = False
    
    # Check files
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"  ✅ {description}: {file_path}")
        else:
            print(f"  ❌ Missing {description}: {file_path}")
            structure_ok = False
    
    return structure_ok

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        "crewai": "CrewAI framework",
        "litellm": "LLM integration", 
        "docx": "Document processing",
        "yaml": "Configuration parsing",
        "click": "CLI interface",
        "structlog": "Structured logging"
    }
    
    deps_ok = True
    installed_packages = []
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            if package == "docx":
                importlib.import_module("docx")
            elif package == "yaml":
                importlib.import_module("yaml")
            else:
                importlib.import_module(package)
            print(f"  ✅ {description}: {package}")
            installed_packages.append(package)
        except ImportError:
            print(f"  ❌ Missing {description}: {package}")
            missing_packages.append(package)
            deps_ok = False
    
    if missing_packages:
        print(f"\n  📦 To install missing packages:")
        if "crewai" in missing_packages:
            print(f"    pip install 'crewai[tools]==0.177.0'")
        if "litellm" in missing_packages:
            print(f"    pip install litellm")
        if "docx" in missing_packages:
            print(f"    pip install python-docx")
        if "yaml" in missing_packages:
            print(f"    pip install PyYAML")
        if "click" in missing_packages:
            print(f"    pip install click")
        if "structlog" in missing_packages:
            print(f"    pip install structlog")
    
    return deps_ok

def check_api_keys():
    """Check API key configuration"""
    print("Checking API keys...")
    
    api_keys = {
        "GEMINI_API_KEY": "Gemini API",
        "GOOGLE_API_KEY": "Google API", 
        "OPENAI_API_KEY": "OpenAI API",
        "ANTHROPIC_API_KEY": "Anthropic API"
    }
    
    configured_keys = []
    
    for key, service in api_keys.items():
        value = os.getenv(key)
        if value and len(value) > 10:  # Basic validation
            print(f"  ✅ {service}: {key} (configured)")
            configured_keys.append(key)
        else:
            print(f"  ❌ {service}: {key} (not set)")
    
    if configured_keys:
        print(f"  🎯 Using: {configured_keys[0]}")
        return True
    else:
        print("  ⚠️  No API keys configured")
        return False

def check_imports():
    """Test critical imports"""
    print("Testing imports...")
    
    import_tests = [
        ("sys", "System module"),
        ("os", "Operating system interface"),
        ("pathlib", "Path handling")
    ]
    
    # Test optional imports
    try:
        import crewai
        import_tests.append(("crewai", "CrewAI framework"))
    except ImportError:
        import_tests.append(("crewai", "CrewAI framework (MISSING)"))
    
    try:
        import litellm
        import_tests.append(("litellm", "LLM integration"))
    except ImportError:
        import_tests.append(("litellm", "LLM integration (MISSING)"))
    
    try:
        from docx import Document
        import_tests.append(("docx", "Document processing"))
    except ImportError:
        import_tests.append(("docx", "Document processing (MISSING)"))
    
    # Test project imports
    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.agent_translator.crew import DocumentTranslationCrew
        import_tests.append(("src.agent_translator.crew", "Translation crew"))
    except ImportError as e:
        import_tests.append(("src.agent_translator.crew", f"Translation crew (MISSING: {e})"))
    
    try:
        from src.agent_translator.tools.custom_tools import DocumentAnalyzerTool
        import_tests.append(("src.agent_translator.tools.custom_tools", "Custom tools"))
    except ImportError as e:
        import_tests.append(("src.agent_translator.tools.custom_tools", f"Custom tools (MISSING: {e})"))
    
    imports_ok = True
    for module, description in import_tests:
        if "MISSING" in description:
            print(f"  ❌ {description}")
            imports_ok = False
        else:
            print(f"  ✅ {description}")
    
    return imports_ok

def run_basic_test():
    """Run a basic functionality test"""
    print("Running basic functionality test...")
    
    try:
        # Test configuration loading
        if Path("config.yaml").exists():
            import yaml
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            print("  ✅ Configuration loading")
        else:
            print("  ❌ Configuration file missing")
            return False
        
        # Test crew initialization
        sys.path.insert(0, str(Path.cwd()))
        from src.agent_translator.crew import DocumentTranslationCrew
        
        crew = DocumentTranslationCrew("config.yaml")
        status = crew.get_translation_status()
        
        print("  ✅ Crew initialization")
        print(f"    System: {status['system']}")
        print(f"    Status: {status['status']}")
        print(f"    Model: {status['llm_model']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic test failed: {e}")
        return False

def provide_fix_suggestions(results):
    """Provide specific fix suggestions based on diagnostic results"""
    print("\n🔧 Fix Suggestions:")
    print("=" * 50)
    
    if not results.get("python_version"):
        print("1. Python Version:")
        print("   • Install Python 3.10 or higher")
        print("   • Update your Python installation")
    
    if not results.get("project_structure"):
        print("2. Project Structure:")
        print("   • Run: python setup.py to create missing directories")
        print("   • Copy the provided files to correct locations:")
        print("     - crew.py → src/agent_translator/crew.py")
        print("     - custom_tools.py → src/agent_translator/tools/custom_tools.py") 
        print("     - main.py → main.py")
    
    if not results.get("dependencies"):
        print("3. Dependencies:")
        print("   • Run: pip install -r requirements.txt")
        print("   • Or install individually:")
        print("     pip install 'crewai[tools]==0.177.0' litellm python-docx PyYAML click structlog")
    
    if not results.get("api_keys"):
        print("4. API Keys:")
        print("   • Set environment variable: export GEMINI_API_KEY='your_key_here'")
        print("   • Or create .env file with: GEMINI_API_KEY=your_key_here")
        print("   • Get API key from: https://makersuite.google.com/app/apikey")
    
    if not results.get("imports"):
        print("5. Import Issues:")
        print("   • Check file locations match expected structure")
        print("   • Verify __init__.py files exist in all packages")
        print("   • Run: python -c 'import sys; print(sys.path)' to check Python path")
    
    if not results.get("basic_test"):
        print("6. Basic Functionality:")
        print("   • Check previous issues first")
        print("   • Verify config.yaml is valid YAML format")
        print("   • Test with: python main.py status")

def main():
    """Run complete diagnostic"""
    print("🔍 CrowdWisdomTrading Translation System Diagnostics")
    print("=" * 60)
    
    results = {}
    
    # Run all diagnostic checks
    results["python_version"] = check_python_version()
    print()
    
    results["project_structure"] = check_project_structure()
    print()
    
    results["dependencies"] = check_dependencies()
    print()
    
    results["api_keys"] = check_api_keys()
    print()
    
    results["imports"] = check_imports()
    print()
    
    results["basic_test"] = run_basic_test()
    print()
    
    # Summary
    print("📊 Diagnostic Summary:")
    print("=" * 60)
    
    checks = [
        ("Python Version", results["python_version"]),
        ("Project Structure", results["project_structure"]),
        ("Dependencies", results["dependencies"]),
        ("API Keys", results["api_keys"]),
        ("Import Tests", results["imports"]),
        ("Basic Functionality", results["basic_test"])
    ]
    
    passed = 0
    failed = 0
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All diagnostics passed! System should be ready.")
        print("Try running: python main.py translate sample_document.docx")
    else:
        provide_fix_suggestions(results)

if __name__ == "__main__":
    main()