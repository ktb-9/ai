# ai


## 패키지 설치
```
!pip install -r requirements.txt
```

## TypeScript 코드
ngrok 서버환경 사용시에는 ngrok 키가 필요합니다. 입력 부분에 키를 넣으세요.

```
import { exec } from "child_process";
import ngrok from "ngrok";

// Flask 서버 실행 함수
function runFlask(): void {
  console.log("Flask 서버를 실행합니다...");
  const flaskProcess = exec("python flask_server.py");

  flaskProcess.stdout?.on("data", (data) => {
    console.log(`Flask: ${data}`);
  });

  flaskProcess.stderr?.on("data", (data) => {
    console.error(`Flask Error: ${data}`);
  });

  flaskProcess.on("close", (code) => {
    console.log(`Flask 서버가 종료되었습니다. 종료 코드: ${code}`);
  });
}

// ngrok 설정 및 실행
async function startNgrok(): Promise<void> {
  try {
    console.log("ngrok을 설정합니다...");
    const publicUrl = await ngrok.connect({
      addr: 5000,
      authtoken: "이부분에 ngrok 키를 입력하세요", // 여기에 ngrok 토큰 추가
    });

    console.log(`공개된 ngrok URL: ${publicUrl}`);
  } catch (error) {
    console.error("ngrok 실행 중 오류가 발생했습니다:", error);
  }
}

// 서버 실행 및 유지
(async function startServer() {
  try {
    // Flask 서버 실행
    runFlask();

    // ngrok 실행
    await startNgrok();

    // 서버 지속 실행
    console.log("서버가 실행 중입니다. 종료하려면 Ctrl+C를 누르세요.");
  } catch (error) {
    console.error("서버 실행 중 오류가 발생했습니다:", error);
  }
})();
```


## 사용 예시

METHOD POST
```
https://dd9c-34-124-204-129.ngrok-free.app/edit-image
```

BODY
```
{
    "image_url": "https://media.4-paws.org/9/c/9/7/9c97c38666efa11b79d94619cc1db56e8c43d430/Molly_006-2829x1886-2726x1886-1920x1328.jpg",
    "instruction": "snowing"
}
```