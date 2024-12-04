from flask import Flask, request, jsonify
import os
import json
from getpass import getpass
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

app = Flask(__name__)

# 환경변수에 OPENAI_API_KEY가 없다면, 사용자로부터 입력을 받습니다.
# 환경 변수를 외부에서 설정할 수 있도록 변경
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = input("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key  # 환경변수로 설정

# 모델 설정
model = ChatOpenAI(
    model_name="gpt-4o-mini",  # GPT 모델 버전 선택
    temperature=0,     
)

# 출력형식을 지정합니다.
class Recommend(BaseModel):
    memberId: int = Field(description="MemberId")
    insights: str = Field(description="Please analyze my consumption habits and give me recommendations.please answer in korean")

# 입력을 JSON으로 받도록 변경
def process_expenses_query(expenses_query_json):
    expenses_query = json.dumps(expenses_query_json)
    parser = JsonOutputParser(pydantic_object=Recommend)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    result = chain.invoke({"query": expenses_query})
    return result

# API 엔드포인트 정의
@app.route('/process_expenses', methods=['POST'])
def process_expenses():
    try:
        # 요청에서 JSON 데이터 받기
        expenses_query_json = request.get_json()

        # 데이터를 처리하여 결과 반환
        output = process_expenses_query(expenses_query_json)
        
        # JSON 형식으로 결과 반환
        return jsonify(output)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
