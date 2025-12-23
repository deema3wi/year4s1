CMM Ops: Mission Control (Windows, WPF, .NET 10)

Build & run (Windows):
  dotnet restore
  dotnet run --project CMM.Ops.MissionControl\CMM.Ops.MissionControl.csproj

Controls:
  - New Mission: regenerates map + base/target
  - Plan Route: A* with Balanced settings
  - Run Monte-Carlo: chooses between Fast/Balanced/Safe route candidates
  - Start Sim / Stop: runs the digital-twin simulation (battery/health/payload) with dynamic obstacles
  - Edit map: left-click toggles obstacles, right-click cycles risk (0 -> 0.35 -> 0.7 -> 1 -> 0)

Notes:
  - This project intentionally excludes Lab 2 (CNN) and Lab 10 (YOLO).
  - Implemented topics: Lab 1 (digital twin), Lab 3 (avoidance), Lab 6 (Monte-Carlo strategy), Lab 7 (logistics), Lab 8 (staff), Lab 9 (routing), Lab 12 (Petri net).
