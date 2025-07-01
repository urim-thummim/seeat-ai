# 씨앗AI 백엔드

씨앗교회의 설교를 기반으로 성도들의 질문에 AI가 답변하는 RAG(Retrieval-Augmented Generation) 서비스의 백엔드 프로젝트입니다.

## 기술 스택

- FastAPI
- OpenAI (GPT-4, Whisper)
- LangChain
- ChromaDB
- yt-dlp
- Kiwipiepy

## 시작하기

1. Python 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 설정을 입력하세요
```

4. 서버 실행:
```bash
uvicorn main:app --reload
```

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
backend/
├── app/
│   ├── api/          # FastAPI 라우터
│   ├── core/         # 설정, 보안
│   ├── services/     # 비즈니스 로직
│   ├── models/       # 데이터 모델
│   └── utils/        # 유틸리티 함수
├── tests/            # 테스트 코드
├── requirements.txt  # 의존성 목록
└── main.py          # 애플리케이션 진입점
```