# 미션 1 · 현재 실행 방식

기본은 챕터 6, 센서로 만드는 온라인 지도, 좌표 입력 연습이다. LLM 없이 실행한다.

```bash
# conda haic_WhereAmI 및 install/setup.bash를 불러온 저장소 최상위 폴더
python src/custom/m1_tools/try_m1.py --goal 16 12
```

다음 시도는 `--goal X Y`만 바꾼다. 화면 없이 실행하려면 `--headless`를 추가한다.

빌드·초기화의 시작점 가정·저장 지도 모드·룰북상 팀 정책 요구는 [README.md](README.md), 실제 검증 결과는 [VALIDATION_M1.md](VALIDATION_M1.md)를 따른다.
