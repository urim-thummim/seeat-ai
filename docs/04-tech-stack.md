# 기술 스택 상세

*최종 업데이트: 2024-07-01*

## 1. 영상 처리
- **유튜브 다운로드**: yt-dlp
- **오디오 추출**: FFmpeg
- **저장소**: 로컬 파일 시스템

## 2. STT (Speech-to-Text)
- **주 모델**: Whisper (로컬 실행)
  - 초기: medium 모델
  - 필요시: large-v3 모델
- **최적화**: 
  - 한국어 특화 설정
  - 배치 처리
  - GPU 가속 (가능시)

## 3. 텍스트 처리
- **전처리**: 
  - LangChain TextSplitter
  - Kiwi (한국어 형태소 분석)
- **청킹**: 
  - RecursiveCharacterTextSplitter
  - 설교 특화 커스텀 로직

## 4. 임베딩
- **모델 옵션**:
  - sentence-transformers
  - ko-sbert 계열
  - (필요시) OpenAI 임베딩
- **최적화**:
  - 배치 처리
  - 캐싱
  - 증분 업데이트

## 5. 벡터 DB
- **개발**: ChromaDB (로컬)
  - 메모리 모드
  - 영구 저장소
- **프로덕션** (필요시):
  - Pinecone 최소 플랜
  - 또는 ChromaDB 유지

## 6. RAG (Retrieval-Augmented Generation)
- **프레임워크**: LangChain
- **검색 전략**:
  - 하이브리드 검색 (의미 + 키워드)
  - 메타데이터 필터링
  - 컨텍스트 최적화

## 7. LLM (Language Model)
- **모델 선택**: 
  - 상황에 따라 적절한 모델 선택
  - 비용과 성능 균형 고려
- **통합**:
  - LangChain
  - 스트리밍 응답
  - 캐싱 시스템

## 8. 백엔드
- **프레임워크**: FastAPI
- **비동기 처리**: asyncio
- **의존성 관리**: Poetry
- **문서화**: FastAPI Swagger

## 9. 프론트엔드
- **프레임워크**: Next.js 14
- **상태 관리**: Zustand
- **스타일링**: TailwindCSS
- **데이터 페칭**: TanStack Query

## 10. 평가 및 모니터링
- **RAG 평가**: RAGAs
- **로깅**: Python logging
- **모니터링**: 기본 메트릭스

## 11. 인프라
- **개발**: 로컬 환경
- **배포** (선택사항):
  - 저비용 VPS
  - Docker
  - Nginx

## 12. 비용 최적화
- 로컬 모델 우선 사용
- 캐싱 적극 활용
- 배치 처리 구현
- API 호출 최적화

## 13. 보안
- 환경 변수 관리
- API 키 보호
- 기본 인증 (필요시)

## 14. 확장성 고려사항
- 모듈식 설계
- 플러그인 아키텍처
- 설정 외부화

## 15. 개발 도구
- **버전 관리**: Git
- **코드 품질**: 
  - black
  - flake8
  - mypy
- **테스트**: pytest

---

**문서 버전**: v1.0  
**최종 업데이트**: 2024-12-28  
**다음 검토 예정**: 2025-01-15 