from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

chat_model = ChatOllama(model="llama3.1:latest")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Answer questions only based on the context provided below. 
If the answer is not contained within the context, respond with "I don't know".

Context: {context}

Question: {question}

Answer:""",
        )
    ]
)

context = """The ALPHA-27 pipeline section was inspected on 12 January 2026 using an inline inspection tool equipped with high-resolution magnetic flux leakage and ultrasonic sensors. The total inspected length was 42 kilometers. Overall pipeline integrity is rated as moderate risk for the next 12 months, given current operating conditions.

Three notable anomaly clusters were detected. Cluster A is located between kilometer 5.2 and 5.9, with a series of external corrosion indications. The deepest metal loss in this cluster is estimated at 34% of wall thickness, with an average depth of 18%. The primary suspected cause is coating degradation combined with insufficient cathodic protection in this area. No immediate leak is expected, but the safety margin is reduced.

Cluster B is located near kilometer 18.7, where internal corrosion features with localized pitting were identified. The deepest pit shows an estimated 28% wall loss, but it is in a low-stress region of the pipeline. The most likely cause is intermittent presence of corrosive fluids due to incomplete dehydration. Current data suggests that standard monitoring is sufficient, but the operator should re-evaluate fluid conditioning procedures.

Cluster C is a group of geometric anomalies around kilometer 31.3, including minor dents and ovalities. None of these features exceed current acceptance criteria. However, one dent coincides with a historical construction weld. Although no cracks were detected, this location should be included in the next scheduled inspection to confirm that no fatigue-related growth is occurring.

Operating pressure during the inspection period ranged between 62 and 68 bar, which is within the designed operating envelope for ALPHA-27. No abnormal pressure fluctuations were recorded. Temperature variations along the inspected segment were minor and are not considered a significant contributing factor to the observed anomalies.

The recommended actions are as follows. First, schedule a detailed field verification for Cluster A within the next six months, including coating repair and an assessment of cathodic protection performance in the affected area. Second, review and, if necessary, optimize dehydration and corrosion inhibitor strategies to reduce the likelihood of further internal corrosion near kilometer 18.7. Third, add the dent–weld interaction at kilometer 31.3 to the high-priority watch list for the next inline inspection run. Finally, maintain current operating pressure limits and monitoring routines, as no evidence suggests an immediate need for pressure derating at this time.
"""

chain = prompt | chat_model

def main():
    print("ALPHA-27 Q&A assistant. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ")
        if question.lower() in {"exit", "quit"}:
            break

        result = chain.invoke({"context": context, "question": question})
        print("Answer:", result.content)

if __name__ == "__main__":
    main()
