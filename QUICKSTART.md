# QUICK START — UAV Traffic AI System

## Two Commands to Run

### Terminal 1: Pipeline & API
```bash
python main.py
```

Wait for: `FastAPI is ready!`

### Terminal 2: Dashboard
```bash
streamlit run dashboard/app.py
```

Wait for: `Running on http://localhost:8501`

Then open: http://localhost:8501

---

## What You'll See

✅ Green status (RUNNING)  
✅ Live annotated video  
✅ Active tracks with class labels  
✅ Collision alerts in real-time  
✅ Agent decisions (MONITOR/HOLD/REROUTE/EMERGENCY_STOP)  
✅ Analytics, reasoning, memory, performance tabs  
✅ FPS counter (15-20 on CPU)  

---

## Stop

Press `Ctrl+C` in each terminal

---

## Issues?

1. **No API connection?** → Make sure main.py is running in Terminal 1
2. **No video frame?** → Wait 3 seconds, main.py is processing first frame
3. **No tracks?** → Check video has vehicles and YOLO loads successfully
4. **Low FPS?** → Normal on CPU; GPU would be 5-10x faster

---

See AUDIT_REPORT.md and FIXES_SUMMARY.md for complete details
