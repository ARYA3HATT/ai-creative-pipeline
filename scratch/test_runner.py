import requests
import time
import os
import zipfile
import json
import sqlite3

BASE_URL = "http://localhost:8000"
DB_PATH = "shared_data/jobs.db"

def query_job_from_db(job_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, status, current_node, retry_count, error_message, metadata_json FROM jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "status": row[1],
            "current_node": row[2],
            "retry_count": row[3],
            "error_message": row[4],
            "metadata": json.loads(row[5]) if row[5] else None
        }
    return None

def wait_for_job(job_id, timeout=1800):
    print(f"Polling status for Job {job_id}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Query DB directly for most accurate live state
        job = query_job_from_db(job_id)
        if job:
            status = job["status"]
            print(f" -> Job Status: {status} | Current Node: {job['current_node']} | Retry Count: {job['retry_count']}")
            if status in ["completed", "failed"]:
                return job
        time.sleep(10)
    raise TimeoutError(f"Job {job_id} exceeded timeout of {timeout} seconds.")

def run_test_1():
    print("\n" + "="*50)
    print("TEST 1: Single URL live generation")
    print("="*50)
    
    url = "https://www.amazon.in/dp/B0CHX1W1XY"
    payload = {"url": url}
    
    print(f"Submitting URL {url} to /api/v1/generate...")
    resp = requests.post(f"{BASE_URL}/api/v1/generate", json=payload)
    if resp.status_code != 200:
        print(f"ERROR: Failed to submit job. Status code: {resp.status_code}. Response: {resp.text}")
        return None
        
    job_id = resp.json()["job_id"]
    print(f"Job enqueued successfully! Job ID: {job_id}")
    
    # Wait for completion
    job = wait_for_job(job_id)
    print(f"Job finished! Status: {job['status']}")
    
    if job["status"] == "failed":
        print(f"ERROR: Job failed with error: {job['error_message']}")
        return job_id
        
    # Download ZIP and confirm contents
    download_url = f"{BASE_URL}/api/v1/job/{job_id}/download"
    print(f"Downloading packaged ZIP from {download_url}...")
    zip_resp = requests.get(download_url)
    if zip_resp.status_code != 200:
        print(f"ERROR: Failed to download ZIP. Status code: {zip_resp.status_code}")
        return job_id
        
    zip_dir = "scratch/extracted_test1"
    os.makedirs(zip_dir, exist_ok=True)
    zip_filepath = os.path.join(zip_dir, "output.zip")
    
    with open(zip_filepath, "wb") as f:
        f.write(zip_resp.content)
        
    print(f"ZIP saved to {zip_filepath}. Extracting...")
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(zip_dir)
        
    # Verify contents
    extracted_files = []
    for root, dirs, files in os.walk(zip_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), zip_dir)
            extracted_files.append(rel_path)
            
    print("Extracted files:")
    for f in extracted_files:
        print(f" - {f}")
        
    pngs = [f for f in extracted_files if f.endswith(".png") and "images/" in f]
    mp4s = [f for f in extracted_files if f.endswith(".mp4") and "videos/" in f]
    
    print(f"Found {len(pngs)} PNGs (expected 5) and {len(mp4s)} MP4s (expected 2).")
    
    # Verify metadata.json
    metadata_json_path = os.path.join(zip_dir, "metadata.json")
    if os.path.exists(metadata_json_path):
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)
        qa_status = metadata.get("qa_status")
        scores_history = metadata.get("critic_scores_history", [])
        print(f"metadata.json qa_status: {qa_status} | scores_history: {scores_history}")
    else:
        print("ERROR: metadata.json not found in ZIP package!")
        
    return job_id

def run_test_2():
    print("\n" + "="*50)
    print("TEST 2: Retry loop verification")
    print("="*50)
    
    url = "https://www.amazon.in/dp/B0CHX1W1XY?retry=true"
    payload = {"url": url}
    
    print(f"Submitting URL {url} to /api/v1/generate...")
    resp = requests.post(f"{BASE_URL}/api/v1/generate", json=payload)
    if resp.status_code != 200:
        print(f"ERROR: Failed to submit job. Status code: {resp.status_code}. Response: {resp.text}")
        return None
        
    job_id = resp.json()["job_id"]
    print(f"Job enqueued successfully! Job ID: {job_id}")
    
    # Wait for completion
    job = wait_for_job(job_id)
    print(f"Job finished! Status: {job['status']}")
    
    # Check database scores history progression
    if job["metadata"]:
        scores = job["metadata"].get("critic_scores", [])
        print(f"Scores progression across attempts: {scores}")
    else:
        print("ERROR: Metadata not present in DB Job Record!")
        
    return job_id

def run_test_3():
    print("\n" + "="*50)
    print("TEST 3: Bulk CSV fault isolation")
    print("="*50)
    
    csv_content = """url
https://www.amazon.in/dp/B0CHX1W1XY
https://this-url-does-not-exist-xyz123.com
https://www.flipkart.com/apple-iphone-15-blue-128-gb/p/itm6ac6485515ae4"""
    
    csv_filepath = "scratch/test_bulk.csv"
    os.makedirs("scratch", exist_ok=True)
    with open(csv_filepath, "w") as f:
        f.write(csv_content)
        
    print(f"CSV file created at {csv_filepath}. Uploading to /api/v1/bulk...")
    files = {"file": ("test_bulk.csv", open(csv_filepath, "rb"), "text/csv")}
    
    resp = requests.post(f"{BASE_URL}/api/v1/bulk", files=files)
    if resp.status_code != 200:
        print(f"ERROR: Bulk submit failed with status: {resp.status_code}. Response: {resp.text}")
        return None
        
    batch_data = resp.json()
    batch_id = batch_data["batch_id"]
    job_ids = batch_data["job_ids"]
    
    print(f"Bulk batch enqueued successfully! Batch ID: {batch_id} | Job IDs: {job_ids}")
    
    # Wait for all enqueued jobs to complete
    completed_jobs = []
    failed_jobs = []
    
    for jid in job_ids:
        job = query_job_from_db(jid)
        if job["status"] in ["completed", "failed"]:
            print(f" -> Job {jid} instantly loaded: Status: {job['status']} | Node: {job['current_node']} | Error: {job['error_message']}")
            if job["status"] == "failed":
                failed_jobs.append(jid)
            else:
                completed_jobs.append(jid)
        else:
            job = wait_for_job(jid)
            if job["status"] == "failed":
                failed_jobs.append(jid)
            else:
                completed_jobs.append(jid)
                
    print(f"Bulk validation finished. Completed: {len(completed_jobs)} | Failed: {len(failed_jobs)}")
    
    # Confirm exact records in DB
    print("\nDatabase batch report:")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, url, status, current_node, error_message FROM jobs WHERE batch_id = ?", (batch_id,))
    rows = cursor.fetchall()
    conn.close()
    
    for r in rows:
        print(f" - Job ID: {r[0]} | URL: {r[1]} | Status: {r[2]} | Current Node: {r[3]} | Error: {r[4]}")
        
    return batch_id

if __name__ == "__main__":
    test1_jid = run_test_1()
    test2_jid = run_test_2()
    batch_id = run_test_3()
