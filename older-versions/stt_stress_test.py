import asyncio
import websockets
import json
import base64
import time
import ssl
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor

class STTStressTest:
    def __init__(self, uri="wss://carriertech.uk:8675", test_audio_path="_2025-03-26_10-16-21.wav"):
        self.uri = uri
        self.test_audio_path = Path(test_audio_path)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Test parameters
        self.concurrent_users = 10  # Number of concurrent connections
        self.requests_per_user = 5  # Number of requests each user will make
        self.total_requests = self.concurrent_users * self.requests_per_user
        
        # Statistics
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []

    def load_test_audio(self):
        """Load and encode test audio file"""
        try:
            with open(self.test_audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                return f"data:audio/wav;base64,{base64.b64encode(audio_data).decode('utf-8')}"
        except FileNotFoundError:
            print(f"Error: Test audio file not found at {self.test_audio_path}")
            return None

    async def single_stt_request(self, user_id):
        """Perform a single STT request"""
        try:
            async with websockets.connect(self.uri, ssl=self.ssl_context) as websocket:
                # Prepare test data
                test_data = {
                    "task": "stt",
                    "blob": self.audio_data,
                    "sentence": "This is a test sentence.",
                    "language": "fr",
                    "username": f"test_user_{user_id}"
                }

                # Record start time
                start_time = time.time()

                # Send request
                await websocket.send(json.dumps(test_data))
                
                # Wait for response
                response = await websocket.recv()
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Process response
                response_data = json.loads(response)
                if "error" not in response_data:
                    self.successful_requests += 1
                    self.response_times.append(response_time)
                    print(f"Request successful - User {user_id} - Time: {response_time:.2f}s")
                else:
                    self.failed_requests += 1
                    print(f"Request failed - User {user_id} - Error: {response_data['error']}")

        except Exception as e:
            self.failed_requests += 1
            print(f"Connection error - User {user_id}: {str(e)}")

    async def user_session(self, user_id):
        """Simulate a user session with multiple requests"""
        for i in range(self.requests_per_user):
            await self.single_stt_request(user_id)
            # Random delay between requests (0.5 to 2 seconds)
            await asyncio.sleep(random.uniform(0.5, 2))

    async def run_stress_test(self):
        """Run the stress test with multiple concurrent users"""
        print(f"Starting stress test with {self.concurrent_users} concurrent users")
        print(f"Each user will make {self.requests_per_user} requests")
        print(f"Total requests planned: {self.total_requests}")
        
        # Load test audio
        self.audio_data = self.load_test_audio()
        if not self.audio_data:
            return
        
        # Create tasks for all users
        tasks = [self.user_session(i) for i in range(self.concurrent_users)]
        
        # Start timing
        start_time = time.time()
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print results
        self.print_results(total_time)

    def print_results(self, total_time):
        """Print test results and statistics"""
        print("\n=== Stress Test Results ===")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Successful requests: {self.successful_requests}")
        print(f"Failed requests: {self.failed_requests}")
        
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            max_response_time = max(self.response_times)
            min_response_time = min(self.response_times)
            
            print(f"\nResponse Times:")
            print(f"Average: {avg_response_time:.2f} seconds")
            print(f"Maximum: {max_response_time:.2f} seconds")
            print(f"Minimum: {min_response_time:.2f} seconds")
        
        print(f"\nRequests per second: {self.successful_requests / total_time:.2f}")

def main():
    # Create and run stress test
    stress_test = STTStressTest()
    asyncio.run(stress_test.run_stress_test())

if __name__ == "__main__":
    main()