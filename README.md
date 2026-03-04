# Cursor CLI to OpenAI API

로컬 PC에서 [Cursor Headless CLI](https://cursor.com/docs/cli/headless)를 **OpenAI 호환** HTTP API로 노출합니다. 다른 앱에서 `base_url=http://localhost:8080/v1` 로 채팅 완료(스트리밍 포함)를 호출할 수 있습니다.

## 요구 사항

- Python 3.10+
- [Cursor CLI](https://cursor.com/docs/cli/installation) 설치 및 인증 (`agent` 명령 사용 가능)
- (선택) `CURSOR_API_KEY` 환경 변수 또는 CLI 로그인

## 설치

```bash
cd CursorCLI2API
pip install -r requirements.txt
```

## 설정

`.env.example` 을 `.env` 로 복사한 뒤 필요 시 수정합니다.

- `HOST`, `PORT`: 서버 바인드 (기본 `127.0.0.1:8080`)
- `CURSOR_AGENT_CMD`: agent 실행 파일 이름/경로 (기본 `agent`)
- `CURSOR_AGENT_CWD`: agent 작업 디렉터리 (비워두면 서버 cwd). **첫 응답이 느릴 때**: 서버를 프로젝트 폴더에서 실행하면 agent가 해당 프로젝트를 인덱싱해 30초 이상 걸릴 수 있음. 프로젝트 컨텍스트가 필요 없으면 `CURSOR_AGENT_CWD=$HOME` 등으로 설정하면 터미널에서 직접 실행할 때처럼 빠르게 응답함.
- `REQUEST_TIMEOUT`: 요청 타임아웃(초), 비우면 무제한
- `CURSOR_AGENT_FORCE`: agent에 `--force` 전달 여부 (기본 `true`, `false`로 끄기)

## 실행

```bash
# 프로젝트 루트에서
uvicorn src.main:app --host 127.0.0.1 --port 8080
```

또는

```bash
python -m src.main
```

서버는 기본적으로 `http://127.0.0.1:8080` 에서만 수신합니다.

## 테스트

- **자동 테스트** (Cursor CLI 없이 mock으로 실행): 프로젝트 루트에서 `pytest tests/ -v`
- **수동 검증**: 서버 실행 후 `python scripts/test_client.py --prompt "Hello"` (기본: 스트리밍). `python scripts/test_client.py --no-stream --prompt "Hello"` 로 비스트리밍/프롬프트 지정 가능.

```powershell
PS > python scripts/test_client.py --prompt "Hello"        
GET http://127.0.0.1:8080/v1/models
Models: {
  "object": "list",
  "data": [
    {
      "id": "cursor-agent",
      "object": "model",
      "created": 0,
      "owned_by": "cursor-cli"
    }
  ]
}

POST http://127.0.0.1:8080/v1/chat/completions (stream=True)
Content:
Waiting for response...

Hello. How can I help you today with your CursorCLI2API project or anything else?
```

## API

- **GET /v1/models**  
  사용 가능 모델 목록. `cursor-agent` 한 개 반환.

- **POST /v1/chat/completions**  
  OpenAI와 동일한 요청 형식.

  - `messages`: 필수. `[{ "role": "user", "content": "..." }]` 형태.
  - `stream`: `true` 이면 SSE 스트리밍, `false` 이면 단일 JSON 응답.
  - `model`: 무시됨 (항상 Cursor agent 사용).

### 예: 비스트리밍

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cursor-agent","messages":[{"role":"user","content":"Say hello in one sentence."}],"stream":false}'
```

### 예: 스트리밍

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cursor-agent","messages":[{"role":"user","content":"Count from 1 to 5."}],"stream":true}'
```

### 다른 앱에서 사용 (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",  # 로컬 전용이라 임의 값
)

r = client.chat.completions.create(
    model="cursor-agent",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in r:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 라이선스

MIT.
