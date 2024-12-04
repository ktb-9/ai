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
  const flaskProcess = exec("python app.py");

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
http://127.0.0.1:5000/process_expenses
```

BODY
```
[
    {
        "paymentId": 1,
        "tripId": 18,
        "category": "술",
        "description": "퍼지 네이블",
        "price": 100000,
        "pay": 1,
        "date": "2024-11-27T12:49:32.000Z",
        "group": [
            {"user_id": 1, "nickname": "사공광열"},
            {"user_id": 2, "nickname": "주연"},
            {"user_id": 3, "nickname": "유정"}
        ]
    },
    {
        "paymentId": 2,
        "tripId": 18,
        "category": "간식",
        "description": "가나 초콜릿",
        "price": 2000,
        "pay": 1,
        "date": "2024-11-27T12:49:32.000Z",
        "group": []
    }
]
```

## 출력 결과
```
{
"insights": "소비 습관을 분석한 결과, 고급 식사와 관련된 지출이 많습니다. 예산을 조정하고, 외식 빈도를 줄이는 것을 고려해 보세요. 또한, 필요하지 않은 지출을 줄이는 것이 좋습니다.",
"memberId": 1
}
```