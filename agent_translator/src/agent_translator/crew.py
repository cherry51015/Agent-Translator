"""
CrowdWisdomTrading Document Translation System - Main Crew
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any

from crewai import Agent, Task, Crew
from crewai.process import Process
from langchain.schema import HumanMessage

from .tools.custom_tools import (
    DocumentAnalyzerTool,
    DocumentStructureTool, 
    BatchTranslatorTool,
    DocxFormatterTool
)

class DocumentTranslationCrew:
    """Main translation crew orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.output_dir = Path(self.config.get("output_directory", "output"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tools
        self.document_analyzer = DocumentAnalyzerTool()
        self.structure_tool = DocumentStructureTool()
        self.translator_tool = BatchTranslatorTool()
        self.formatter_tool = DocxFormatterTool()
        
        # Initialize agents
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        self.crew = self._create_crew()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "llm_model": "gemini/gemini-1.5-flash",
            "temperature": 0.3,
            "source_language": "English", 
            "target_language": "Spanish",
            "max_tokens_per_batch": 2000,
            "output_directory": "output",
            "preserve_formatting": True
        }
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents for translation workflow"""
        
        # Document Analyzer Agent
        document_analyzer = Agent(
            role="Document Analyzer",
            goal="Analyze academic documents to understand their type, field, language, and structure",
            backstory="You are a specialized academic document analyst with expertise in identifying document types, academic fields, and structural elements across multiple languages and disciplines.",
            tools=[self.document_analyzer],
            verbose=True,
            allow_delegation=False,
            llm_config={
                "model": self.config["llm_model"],
                "temperature": self.config["temperature"]
            }
        )
        
        # Structure Breakdown Agent  
        structure_agent = Agent(
            role="Structure Analyst",
            goal="Break down document structure and create intelligent translation batches",
            backstory="You are an expert in document structure analysis, specializing in preserving formatting while creating optimal batches for translation processing.",
            tools=[self.structure_tool],
            verbose=True,
            allow_delegation=False,
            llm_config={
                "model": self.config["llm_model"],
                "temperature": self.config["temperature"]
            }
        )
        
        # Academic Translator Agent
        translator_agent = Agent(
            role="Academic Translator", 
            goal=f"Translate academic content from {self.config['source_language']} to {self.config['target_language']} with precision and academic quality",
            backstory=f"You are a professional academic translator specializing in {self.config['target_language']} translations. You maintain academic tone, preserve technical terminology, and ensure conceptual accuracy.",
            tools=[self.translator_tool],
            verbose=True,
            allow_delegation=False,
            llm_config={
                "model": self.config["llm_model"],
                "temperature": self.config["temperature"]
            }
        )
        
        # Document Formatter Agent
        formatter_agent = Agent(
            role="Document Formatter",
            goal="Reconstruct translated content into properly formatted DOCX documents",
            backstory="You are a document formatting specialist who recreates professional academic documents while preserving original structure and applying proper formatting.",
            tools=[self.formatter_tool],
            verbose=True,
            allow_delegation=False,
            llm_config={
                "model": self.config["llm_model"],
                "temperature": self.config["temperature"]
            }
        )
        
        return {
            "document_analyzer": document_analyzer,
            "structure_agent": structure_agent, 
            "translator_agent": translator_agent,
            "formatter_agent": formatter_agent
        }
    
    def _create_tasks(self) -> Dict[str, Task]:
        """Create tasks for the translation workflow"""
        
        analyze_task = Task(
            description="""
            Analyze the provided academic document to determine:
            1. Document type and academic field
            2. Source language and complexity level
            3. Special elements present (tables, figures, citations, etc.)
            4. Field-specific translation requirements
            
            Use the DocumentAnalyzerTool to perform comprehensive analysis.
            Return detailed analysis in JSON format.
            """,
            agent=self.agents["document_analyzer"],
            expected_output="JSON analysis containing document_type, academic_field, source_language, complexity_level, special_elements, and translation_instructions"
        )
        
        structure_task = Task(
            description="""
            Break down the document structure and create intelligent translation batches:
            1. Extract all paragraphs with formatting information
            2. Create translation batches within token limits
            3. Preserve paragraph ordering and relationships
            4. Generate formatting map for reconstruction
            
            Use the DocumentStructureTool with the document path.
            Return structured data for batch processing.
            """,
            agent=self.agents["structure_agent"],
            expected_output="JSON structure containing translation_batches, formatting_map, and total_paragraphs"
        )
        
        translation_task = Task(
            description=f"""
            Translate all document content from {self.config['source_language']} to {self.config['target_language']}:
            1. Process each batch with academic quality translation
            2. Maintain technical terminology appropriately
            3. Preserve academic tone and style
            4. Keep paragraph markers and structure intact
            
            Use the BatchTranslatorTool with structural analysis and target language.
            Return completed translations for all batches.
            """,
            agent=self.agents["translator_agent"],
            expected_output=f"JSON containing all translated content in {self.config['target_language']} with preserved structure"
        )
        
        formatting_task = Task(
            description="""
            Reconstruct the translated content into a properly formatted DOCX document:
            1. Apply original formatting to translated text
            2. Recreate document structure and layout
            3. Preserve tables, lists, and special elements
            4. Generate final formatted output file
            
            Use the DocxFormatterTool with translated content and structural data.
            Return final document creation confirmation.
            """,
            agent=self.agents["formatter_agent"],
            expected_output="JSON confirmation of document creation with output file path and quality metrics"
        )
        
        return {
            "analyze_task": analyze_task,
            "structure_task": structure_task,
            "translation_task": translation_task, 
            "formatting_task": formatting_task
        }
    
    def _create_crew(self) -> Crew:
        """Create the main crew with sequential process"""
        return Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            process=Process.sequential,
            verbose=True
        )
    
    def translate_document(self, document_path: str, output_filename: str = None) -> Dict[str, Any]:
        """Execute the complete translation workflow"""
        try:
            # Validate input
            if not os.path.exists(document_path):
                return {"status": "error", "error": f"Document not found: {document_path}"}
            
            # Generate output filename if not provided
            if not output_filename:
                input_name = Path(document_path).stem
                target_lang = self.config['target_language'].lower().replace(' ', '_')
                output_filename = f"{input_name}_{target_lang}.docx"
            
            output_path = str(self.output_dir / output_filename)
            
            # Prepare inputs for the crew
            inputs = {
                "document_path": document_path,
                "target_language": self.config['target_language'],
                "output_path": output_path,
                "max_tokens_per_batch": self.config['max_tokens_per_batch']
            }
            
            # Execute the crew workflow
            result = self.crew.kickoff(inputs=inputs)
            
            return {
                "status": "success",
                "output_file": output_filename,
                "full_path": output_path,
                "result": str(result)
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "output_file": None
            }
    
    def get_translation_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "system": "CrowdWisdomTrading AI Translation",
            "version": "2.0.0",
            "status": "ready",
            "llm_model": self.config["llm_model"],
            "source_language": self.config["source_language"],
            "target_language": self.config["target_language"],
            "agents": {
                "document_analyzer": "ready",
                "structure_analyst": "ready", 
                "academic_translator": "ready",
                "document_formatter": "ready"
            }
        }