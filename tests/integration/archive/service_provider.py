import asyncio
from src.agent.ea_assistant import ProductionEAAgent
   
agent = ProductionEAAgent()
response = asyncio.run(agent.process_query(
   "What is the definition of a Service Provider?"
))
   
print(response.response)
print(f"\nCitations: {response.citations}")
   
# Validate
from src.llm.prompts import EAPromptTemplate
validation = EAPromptTemplate.validate_response_format(response.response)
print(f"\nValidation: {validation['verdict']}")
print(f"Issues: {validation['issues']}")