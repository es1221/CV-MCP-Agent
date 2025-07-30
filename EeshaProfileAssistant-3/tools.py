import os
import json
import logging
from typing import List, Dict, Any
from embed import DocumentEmbedder

class CVTools:
    def __init__(self):
        """Initialize CV tools with document embedder"""
        self.embedder = DocumentEmbedder()
        self.repo_summaries = self._load_repo_summaries()
        
    def _load_repo_summaries(self) -> Dict[str, str]:
        """Load GitHub repository summaries from JSON file"""
        try:
            with open('data/repo_summaries.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("repo_summaries.json not found, using empty dict")
            return {}
        except Exception as e:
            logging.error(f"Error loading repo summaries: {str(e)}")
            return {}
    
    def retrieve_background(self, query: str) -> str:
        """
        Search for information about Eesha's background using vector similarity
        
        Args:
            query: Search query about Eesha's background
            
        Returns:
            Relevant information from CV, dissertation, and project notes
        """
        try:
            # Search for relevant documents
            results = self.embedder.search(query, top_k=3)
            
            if not results:
                return "No relevant information found in the documents."
            
            # Combine results with context
            response = "Based on the available information:\n\n"
            
            for i, (text, score, source) in enumerate(results, 1):
                if score > 0.3:  # Include moderate to high-confidence results
                    response += f"{i}. From {source}: {text}\n\n"
            
            if response == "Based on the available information:\n\n":
                return "No highly relevant information found for this query."
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Error in retrieve_background: {str(e)}")
            return "I encountered an error while searching for information."
    
    def summarise_repo(self, repo_name: str) -> str:
        """
        Get summary of a GitHub repository

        Args:
            repo_name: Name of the repository

        Returns:
            Repository summary or error message
        """
        try:
            clean_repo_name = repo_name.lower().strip()

            for repo_key, info in self.repo_summaries.items():
                if clean_repo_name in repo_key.lower():
                    if isinstance(info, dict):
                        summary = info.get("summary", "No summary available.")
                        stack = ", ".join(info.get("stack", []))
                        visibility = "Private" if info.get("private", False) else "Public"
                        github_url = info.get("github_url", "")
                    else:
                        summary = "No summary available."
                        stack = ""
                        visibility = "Unknown"
                        github_url = ""

                    github_link = f"\n\n🔗 **GitHub Repository:** {github_url}" if github_url else ""
                    privacy_note = " (Private Repository)" if info.get("private", False) else ""
                    
                    return f"""**{repo_key}**{privacy_note}

{summary}

**Tech Stack:** {stack if stack else 'Not specified'}

**Features:**
{chr(10).join(['• ' + feature for feature in info.get('features', ['No features listed'])][:5])}{github_link}"""
            
            return f"Repository '{repo_name}' not found in the available summaries."

        except Exception as e:
            logging.error(f"Error in summarise_repo: {str(e)}")
            return "I encountered an error while retrieving repository information."
    
    def get_all_projects(self) -> str:
        """
        Get comprehensive information about all projects including GitHub links
        
        Returns:
            Detailed information about all projects with GitHub links
        """
        if not self.repo_summaries:
            return "No repository information available."

        projects = []
        for repo_name in self.repo_summaries.keys():
            project_details = self.summarise_repo(repo_name)
            projects.append(project_details)
        
        return "Here are Eesha's documented projects:\n\n" + "\n\n---\n\n".join(projects)
    
    def get_education(self) -> str:
        """
        Get information about Eesha's educational background
        
        Returns:
            Educational background information
        """
        try:
            # Search for education-related information with better queries
            education_queries = [
                "Imperial College London Design Engineering MEng degree",
                "education university college graduation",
                "Lady Eleanor Holles School A-Level"
            ]
            
            all_results = []
            for query in education_queries:
                results = self.embedder.search(query, top_k=3)
                all_results.extend(results)
            
            if not all_results:
                return "No educational information found in the documents."
            
            # Sort by relevance and remove duplicates with lower threshold
            unique_results = []
            seen_texts = set()
            
            for text, score, source in all_results:
                if text not in seen_texts and score > 0.3:  # Lowered threshold from 0.6 to 0.3
                    unique_results.append((text, score, source))
                    seen_texts.add(text)
            
            # Sort by score
            unique_results.sort(key=lambda x: x[1], reverse=True)
            
            if not unique_results:
                return "Educational background information not found with sufficient confidence."
            
            response = "Educational Background:\n\n"
            
            for i, (text, score, source) in enumerate(unique_results[:3], 1):
                response += f"{i}. {text}\n\n"
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Error in get_education: {str(e)}")
            return "I encountered an error while retrieving educational information."
    
    def get_degree_overview(self) -> str:
        """
        Get information about Eesha's degree and what design engineering entails
        
        Returns:
            Overview of design engineering degree and skills
        """
        try:
            with open('data/deseng.json', 'r') as f:
                data = json.load(f)
            return data["degree_overview"]["description"]
        except Exception as e:
            logging.error(f"Error in get_degree_overview: {str(e)}")
            return "I couldn't retrieve information about Eesha's degree at the moment."

    def list_achievements(self) -> str:
        """
        List Eesha's personal achievements, extracurriculars, and leadership experience
        
        Returns:
            Personal achievements and extracurricular activities
        """
        try:
            with open('data/personal.json', 'r') as f:
                achievements = json.load(f)

            response = "Here are some of Eesha's personal achievements:\n\n"
            for key, entry in achievements.items():
                response += f"{entry['title']}:\n{entry['description']}\n\n"

            return response.strip()
        
        except Exception as e:
            logging.error(f"Error in list_achievements: {str(e)}")
            return "I couldn't retrieve Eesha's achievements right now."
    
    def get_skills(self) -> str:
        """
        Get information about Eesha's technical skills, programming languages, and technologies
        
        Returns:
            Skills and technical proficiencies information
        """
        try:
            # Search for skills-related information with better queries
            skills_queries = [
                "Programming Languages Python HTML CSS JavaScript MATLAB",
                "Libraries Frameworks PyTorch TensorFlow scikit-learn pandas NumPy",
                "technical skills proficiency software tools AI automation",
                "Machine Learning Deep Learning data modelling"
            ]
            
            all_results = []
            for query in skills_queries:
                results = self.embedder.search(query, top_k=3)
                all_results.extend(results)
            
            if not all_results:
                return "No skills information found in the documents."
            
            # Sort by relevance and remove duplicates with lower threshold
            unique_results = []
            seen_texts = set()
            
            for text, score, source in all_results:
                if text not in seen_texts and score > 0.3:  # Lowered threshold
                    unique_results.append((text, score, source))
                    seen_texts.add(text)
            
            # Sort by score
            unique_results.sort(key=lambda x: x[1], reverse=True)
            
            if not unique_results:
                return "Technical skills information not found with sufficient confidence."
            
            response = "Technical Skills and Proficiencies:\n\n"
            
            for i, (text, score, source) in enumerate(unique_results[:4], 1):
                response += f"{i}. {text}\n\n"
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Error in get_skills: {str(e)}")
            return "I encountered an error while retrieving skills information."
    
    def get_personal_info(self) -> str:
        """
        Get personal information about Eesha including contact details and basic info
        
        Returns:
            Personal information and contact details
        """
        try:
            # Search for personal/contact information
            personal_queries = [
                "contact email phone location address",
                "personal details name location",
                "professional contact information"
            ]
            
            all_results = []
            for query in personal_queries:
                results = self.embedder.search(query, top_k=2)
                all_results.extend(results)
            
            if not all_results:
                return "No personal information found in the documents."
            
            # Sort by relevance and remove duplicates
            unique_results = []
            seen_texts = set()
            
            for text, score, source in all_results:
                if text not in seen_texts and score > 0.5:
                    unique_results.append((text, score, source))
                    seen_texts.add(text)
            
            # Sort by score
            unique_results.sort(key=lambda x: x[1], reverse=True)
            
            response = "Personal Information:\n\n"
            
            for i, (text, score, source) in enumerate(unique_results[:3], 1):
                response += f"{i}. {text}\n\n"
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Error in get_personal_info: {str(e)}")
            return "I encountered an error while retrieving personal information."

    def get_butterfly_air_info(self, query: str = "") -> str:
        """
        Get comprehensive information about Eesha's work at Butterfly Air
        
        Returns:
            Detailed information about her role, projects, and achievements at Butterfly Air
        """
        try:
            butterfly_info = {
                "role": "AI Solutions Engineer and Product Manager",
                "duration": "March 2024 - June 2025",
                "key_achievements": [
                    "Independently developed a custom AI-driven time series forecasting model using NeuralProphet",
                    "Fused autoregressive methods with neural networks to predict indoor air quality metrics",
                    "Achieved ~94% accuracy over a 12-hour forecast window for IAQ prediction",
                    "Collaborated with cross-functional teams including software developers",
                    "Designed and implemented a language model (LLM) using ChatGPT to analyze forecast data",
                    "Built an AI Agent using Retrieval Augmented Generation model implemented through LangChain",
                    "Achieved zero hallucinations in the AI Agent implementation",
                    "Conducted evaluation using RAGAS framework and k-fold cross-validation",
                    "Positioned Butterfly Air distinctively in the IAQ field through innovative solutions"
                ],
                "technical_details": [
                    "Used NeuralProphet for time series forecasting",
                    "Implemented RAG-based AI Agent with LangChain",
                    "Applied ChatGPT for actionable insights from forecast data",
                    "Conducted client surveys confirming utility of features",
                    "Created product touchpoint maps for current and future clients",
                    "Worked with Butterfly Air's Morpho device for IAQ monitoring",
                    "Monitored parameters: CO2, PM levels, VOC, temperature, humidity"
                ],
                "business_impact": [
                    "First implementation of IAQ prediction technology in the industry",
                    "First implementation of personalized IAQ AI Agent technology",
                    "Differentiated Butterfly Air from competitors like Kaiterra and Awair",
                    "Established business value through innovation and novelty",
                    "Confirmed client utility through surveys and feedback",
                    "Mapped product utility for office clients and potential hospital clients"
                ]
            }
            
            response = f"Eesha Sondhi worked at Butterfly Air as an {butterfly_info['role']} from {butterfly_info['duration']}.\n\n"
            
            response += "Key Achievements:\n"
            for achievement in butterfly_info['key_achievements']:
                response += f"• {achievement}\n"
            
            response += "\nTechnical Implementation:\n"
            for detail in butterfly_info['technical_details']:
                response += f"• {detail}\n"
            
            response += "\nBusiness Impact:\n"
            for impact in butterfly_info['business_impact']:
                response += f"• {impact}\n"
            
            return response
            
        except Exception as e:
            logging.error(f"Error in get_butterfly_air_info: {str(e)}")
            return "I encountered an error while retrieving Butterfly Air information."

    def fallback_response(self) -> str:
        """
        Fallback response when information is not available
        
        Returns:
            Standard fallback message
        """
        return "I don't have specific information about that in my current knowledge base. I can only provide information based on Eesha's CV, dissertation, and documented projects. Please try asking about her work experience, education, or specific projects that might be documented."
