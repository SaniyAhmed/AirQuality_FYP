import ee

# 1. Trigger the browser authentication flow
print("Starting Earth Engine authentication...")
ee.Authenticate()

# 2. Initialize using your registered project ID
project_id = "green-cover-karachi"

try:
    ee.Initialize(project=project_id)
    print(f"\n--- SUCCESS: CONNECTED TO GEE PROJECT '{project_id}' ---")
except Exception as e:
    print(f"\nInitialization failed: {e}")