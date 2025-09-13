#!/usr/bin/env python3
"""
CrowdWisdomTrading Document Translation System - CLI Interface
Fixed version for proper project structure
"""

import os
import sys
import click
import yaml
from pathlib import Path
from datetime import datetime
import structlog

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.agent_translator.crew import DocumentTranslationCrew
except ImportError:
    try:
        # Try alternative path
        from agent_translator.crew import DocumentTranslationCrew
    except ImportError:
        print("❌ Error: Cannot import DocumentTranslationCrew")
        print("Make sure crew.py is in the correct location")
        print("Expected location: src/agent_translator/crew.py")
        sys.exit(1)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@click.group()
@click.version_option(version='2.0.0', prog_name='CrowdWisdomTrading AI Translator')
def cli():
    """
    CrowdWisdomTrading AI Document Translation System
    
    Advanced academic document translation using CrewAI framework.
    Maintains formatting, academic tone, and document structure.
    """
    pass

@cli.command()
@click.argument('document_path', type=click.Path(exists=True))
@click.option('--target-language', '-t', default='Spanish', 
              help='Target language for translation (default: Spanish)')
@click.option('--output-filename', '-o', default=None,
              help='Output filename (auto-generated if not provided)')
@click.option('--config', '-c', default='config.yaml',
              help='Configuration file path (default: config.yaml)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def translate(document_path, target_language, output_filename, config, verbose):
    """
    Translate an academic document while preserving formatting and structure.
    
    DOCUMENT_PATH: Path to the input DOCX document to translate
    """
    try:
        click.echo("🚀 Starting CrowdWisdomTrading AI Translation System")
        click.echo(f"📄 Document: {document_path}")
        click.echo(f"🌐 Target Language: {target_language}")
        click.echo("=" * 60)
        
        # Validate input file
        if not document_path.endswith(('.docx', '.doc')):
            click.echo("❌ Error: Only DOCX and DOC files are supported", err=True)
            sys.exit(1)
        
        # Load/create configuration
        config_data = load_or_create_config(config, target_language)
        
        # Check API keys
        api_key_check = check_api_keys()
        if not api_key_check['has_key']:
            click.echo(f"❌ Error: {api_key_check['message']}")
            sys.exit(1)
        
        click.echo(f"✅ Using API: {api_key_check['key_type']}")
        
        # Initialize translation crew
        click.echo("🤖 Initializing AI Translation Crew...")
        try:
            crew = DocumentTranslationCrew(config)
            
            # Display system status
            status = crew.get_translation_status()
            click.echo(f"✅ System Status: {status['status']}")
            click.echo(f"🧠 LLM Model: {status['llm_model']}")
            click.echo(f"🎯 Target Language: {status['target_language']}")
            
        except Exception as init_error:
            click.echo(f"❌ Failed to initialize crew: {str(init_error)}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        
        # Execute translation with progress indication
        click.echo("\n🔄 Starting Translation Workflow...")
        
        try:
            with click.progressbar(length=100, label='Translation Progress') as bar:
                click.echo("\n📋 Phase 1: Document Analysis...")
                bar.update(25)
                
                click.echo("🔍 Phase 2: Structure Breakdown...")
                bar.update(25)
                
                click.echo("🌐 Phase 3: Academic Translation...")
                bar.update(25)
                
                result = crew.translate_document(document_path, output_filename)
                
                click.echo("📝 Phase 4: Document Formatting...")
                bar.update(25)
        
        except Exception as translation_error:
            click.echo(f"\n❌ Translation Error: {str(translation_error)}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        
        # Handle results
        if result.get('status') == 'success':
            click.echo("\n" + "=" * 60)
            click.echo("✅ TRANSLATION COMPLETED SUCCESSFULLY!")
            click.echo("=" * 60)
            click.echo(f"📁 Output File: {result['output_file']}")
            click.echo(f"📍 Full Path: {result['full_path']}")
            click.echo(f"🎯 Target Language: {target_language}")
            click.echo(f"⏱️  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Verify output file
            if os.path.exists(result['full_path']):
                file_size = os.path.getsize(result['full_path']) / (1024 * 1024)
                click.echo(f"📊 File Size: {file_size:.2f} MB")
                click.echo("\n🎉 Your translated document is ready!")
            else:
                click.echo("⚠️  Warning: Output file not found at expected location")
                
        else:
            click.echo("\n❌ TRANSLATION FAILED")
            click.echo(f"Error: {result.get('error', 'Unknown error occurred')}")
            
            # Provide troubleshooting tips
            click.echo("\n🔧 Troubleshooting suggestions:")
            click.echo("1. Check your API key is valid and has credits")
            click.echo("2. Verify the input document is not corrupted") 
            click.echo("3. Try with a smaller document first")
            click.echo("4. Run 'python main.py test' to check system status")
            
            if verbose and 'error' in result:
                click.echo(f"Detailed error: {result['error']}")
            
            sys.exit(1)
            
    except KeyboardInterrupt:
        click.echo("\n❌ Translation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Critical error in translation", error=str(e))
        click.echo(f"❌ Critical Error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--target-language', '-t', default='Spanish',
              help='Default target language')
def init_config(target_language):
    """Initialize configuration file with default settings"""
    try:
        config_data = {
            "source_language": "English",
            "target_language": target_language,
            "llm_model": "gemini/gemini-1.5-flash",
            "temperature": 0.3,
            "max_tokens_per_batch": 2000,
            "preserve_formatting": True,
            "output_format": "docx",
            "batch_processing": True,
            "output_directory": "output",
            "create_backup": True,
            "overwrite_existing": False,
            "translation_settings": {
                "maintain_academic_tone": True,
                "preserve_citations": True,
                "handle_technical_terms": True,
                "keep_formatting_markers": True,
                "translate_table_headers": True,
                "preserve_footnotes": True
            },
            "token_management": {
                "smart_batching": True,
                "max_words_per_batch": 500,
                "overlap_sentences": 2,
                "retry_on_failure": 3
            },
            "logging": {
                "level": "INFO",
                "file": "translation_system.log",
                "console_output": True
            },
            "performance": {
                "timeout_seconds": 300,
                "retry_attempts": 3,
                "delay_between_requests": 1,
                "max_concurrent_requests": 1
            }
        }
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        click.echo("✅ Configuration file 'config.yaml' created successfully!")
        click.echo(f"🎯 Default target language: {target_language}")
        click.echo("📝 Edit config.yaml to customize translation settings")
        
    except Exception as e:
        click.echo(f"❌ Failed to create configuration: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
def status():
    """Display system status and configuration"""
    try:
        click.echo("🚀 CrowdWisdomTrading AI Translation System Status")
        click.echo("=" * 60)
        
        # Check project structure
        files_to_check = {
            'config.yaml': 'Configuration File',
            'src/agent_translator/crew.py': 'Main Crew File', 
            'src/agent_translator/tools/custom_tools.py': 'Custom Tools',
            'src/agent_translator/__init__.py': 'Package Init'
        }
        
        click.echo("📁 Project Structure:")
        for file_path, description in files_to_check.items():
            exists = os.path.exists(file_path)
            status_icon = "✅" if exists else "❌"
            click.echo(f"  • {description}: {status_icon} {file_path}")
        
        # Check API keys
        api_status = check_api_keys()
        click.echo(f"\n🔑 API Keys: {api_status['message']}")
        
        # Test crew import
        try:
            from src.agent_translator.crew import DocumentTranslationCrew
            click.echo("🤖 Crew Import: ✅ Success")
            
            if os.path.exists('config.yaml'):
                crew = DocumentTranslationCrew('config.yaml')
                sys_status = crew.get_translation_status()
                
                click.echo(f"\n📊 System: {sys_status['system']}")
                click.echo(f"🔢 Version: {sys_status['version']}")
                click.echo(f"⚡ Status: {sys_status['status']}")
                click.echo(f"🧠 LLM Model: {sys_status['llm_model']}")
                click.echo(f"🌐 Target Language: {sys_status['target_language']}")
                
                click.echo("\n🤖 Agent Status:")
                for agent, stat in sys_status['agents'].items():
                    click.echo(f"  • {agent.replace('_', ' ').title()}: {stat}")
                    
        except Exception as e:
            click.echo(f"🤖 Crew Import: ❌ Failed - {str(e)}")
        
        # Check output directory
        output_exists = os.path.exists('output')
        click.echo(f"\n📁 Output Directory: {'✅ Ready' if output_exists else '❌ Missing'}")
        
        if not os.path.exists('config.yaml'):
            click.echo("\n💡 Run 'python main.py init-config' to create configuration")
                
    except Exception as e:
        click.echo(f"❌ Error checking status: {str(e)}", err=True)

@cli.command()
def test():
    """Run comprehensive system tests"""
    click.echo("🧪 Running CrowdWisdomTrading Translation System Tests...")
    click.echo("=" * 60)
    
    tests = []
    
    # Test 1: Configuration
    config_exists = os.path.exists('config.yaml')
    tests.append(("Configuration File", config_exists))
    
    # Test 2: Project Structure
    crew_exists = os.path.exists('src/agent_translator/crew.py')
    tests.append(("Crew Module", crew_exists))
    
    tools_exist = os.path.exists('src/agent_translator/tools/custom_tools.py')
    tests.append(("Tools Module", tools_exist))
    
    # Test 3: Dependencies
    try:
        import crewai
        crewai_ok = True
    except ImportError:
        crewai_ok = False
    tests.append(("CrewAI Library", crewai_ok))
    
    try:
        import litellm
        litellm_ok = True  
    except ImportError:
        litellm_ok = False
    tests.append(("LiteLLM Library", litellm_ok))
    
    try:
        from docx import Document
        docx_ok = True
    except ImportError:
        docx_ok = False
    tests.append(("Python-DOCX Library", docx_ok))
    
    # Test 4: API Keys
    api_status = check_api_keys()
    tests.append(("API Key Configuration", api_status['has_key']))
    
    # Test 5: Crew Import
    crew_import_ok = False
    try:
        from src.agent_translator.crew import DocumentTranslationCrew
        crew_import_ok = True
    except Exception:
        pass
    tests.append(("Crew Class Import", crew_import_ok))
    
    # Test 6: Output Directory
    output_dir = os.path.exists('output')
    tests.append(("Output Directory", output_dir))
    
    # Run all tests
    passed = 0
    failed = 0
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        click.echo(f"Testing {test_name}... {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    # Summary
    click.echo("\n" + "=" * 60)
    click.echo(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        click.echo("🎉 All tests passed! System ready for translation.")
    else:
        click.echo("⚠️  Some tests failed. Check the issues above.")
        
        # Provide setup instructions
        if not config_exists:
            click.echo("\n💡 Run: python main.py init-config")
        if not api_status['has_key']:
            click.echo("💡 Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        if not crewai_ok:
            click.echo("💡 Run: pip install crewai[tools]")
        if not output_dir:
            click.echo("💡 Creating output directory...")
            os.makedirs('output', exist_ok=True)

def check_api_keys():
    """Check for required API keys"""
    # Check for Gemini/Google API keys
    gemini_key = os.getenv('GEMINI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if gemini_key:
        return {"has_key": True, "key_type": "GEMINI_API_KEY", "message": "✅ Gemini API key found"}
    elif google_key:
        return {"has_key": True, "key_type": "GOOGLE_API_KEY", "message": "✅ Google API key found"}
    elif openai_key:
        return {"has_key": True, "key_type": "OPENAI_API_KEY", "message": "✅ OpenAI API key found"}
    elif anthropic_key:
        return {"has_key": True, "key_type": "ANTHROPIC_API_KEY", "message": "✅ Anthropic API key found"}
    else:
        return {
            "has_key": False, 
            "key_type": None, 
            "message": "❌ No API keys found. Set GEMINI_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY"
        }

def load_or_create_config(config_path: str, target_language: str) -> dict:
    """Load existing config or create default one"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Update target language if different
        if target_language != 'Spanish':
            config['target_language'] = target_language
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        return config
    else:
        click.echo(f"⚠️  Config file not found. Creating default: {config_path}")
        default_config = {
            "source_language": "English",
            "target_language": target_language,
            "llm_model": "gemini/gemini-1.5-flash",
            "temperature": 0.3,
            "max_tokens_per_batch": 2000,
            "output_directory": "output",
            "preserve_formatting": True
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        return default_config

if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Check basic requirements before running
    api_status = check_api_keys()
    if not api_status['has_key']:
        print("⚠️  Warning: No API keys found")
        print("Please set one of: GEMINI_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
        print("You can continue to run setup commands, but translation will fail without API keys.")
    
    cli()