import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
import whisper
from anthropic import Anthropic
import logging
from pathlib import Path
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_processor.log"),
        logging.StreamHandler()
    ]
)

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Transcription & Summarization Tool")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)
        
        # Initialize variables
        self.selected_folder = tk.StringVar()
        self.processing_status = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.current_file = tk.StringVar(value="")
        self.whisper_model = None
        self.anthropic_client = None
        self.api_key = tk.StringVar()
        
        # Check for CUDA availability
        self.cuda_available = torch.cuda.is_available()
        self.device_info = f"Using: {'CUDA' if self.cuda_available else 'CPU'}"
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # API key section
        api_frame = ttk.LabelFrame(main_frame, text="API Configuration", padding=10)
        api_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(api_frame, text="Anthropic API Key:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        api_entry = ttk.Entry(api_frame, textvariable=self.api_key, width=50, show="*")
        api_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(api_frame, text=self.device_info).grid(row=0, column=2, sticky=tk.E, padx=5, pady=5)
        
        # Folder selection section
        folder_frame = ttk.LabelFrame(main_frame, text="Folder Selection", padding=10)
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(folder_frame, text="Select Folder:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(folder_frame, textvariable=self.selected_folder, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(folder_frame, text="Browse...", command=self.browse_folder).grid(row=0, column=2, sticky=tk.E, padx=5, pady=5)
        
        # Processing options
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.whisper_model_var = tk.StringVar(value="base")
        ttk.Label(options_frame, text="Whisper Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        whisper_models = ["tiny", "base", "small", "medium", "large"]
        whisper_dropdown = ttk.Combobox(options_frame, textvariable=self.whisper_model_var, values=whisper_models, state="readonly")
        whisper_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Process button
        ttk.Button(main_frame, text="Start Processing", command=self.start_processing).pack(fill=tk.X, padx=5, pady=10)
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Processing Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(status_frame, textvariable=self.processing_status).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(status_frame, text="Current File:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(status_frame, textvariable=self.current_file).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Make grid columns expandable
        main_frame.columnconfigure(0, weight=1)
        folder_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(1, weight=1)
        
    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.selected_folder.set(folder_path)
            self.log(f"Selected folder: {folder_path}")
    
    def log(self, message):
        logging.info(message)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_processing(self):
        folder_path = self.selected_folder.get()
        api_key = self.api_key.get()
        
        if not folder_path:
            messagebox.showerror("Error", "Please select a folder.")
            return
            
        if not api_key:
            messagebox.showerror("Error", "Please enter your Anthropic API key.")
            return
        
        # Start processing in a separate thread
        threading.Thread(target=self.process_folder, args=(folder_path, api_key), daemon=True).start()
    
    def process_folder(self, folder_path, api_key):
        try:
            # Initialize the Anthropic client
            self.anthropic_client = Anthropic(api_key=api_key)
            
            # Load the Whisper model
            whisper_model_name = self.whisper_model_var.get()
            self.processing_status.set(f"Loading Whisper model ({whisper_model_name})...")
            self.log(f"Loading Whisper model: {whisper_model_name}")
            
            device = "cuda" if self.cuda_available else "cpu"
            self.whisper_model = whisper.load_model(whisper_model_name, device=device)
            
            # Get all video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            video_files = [f for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in video_extensions)]
            
            if not video_files:
                self.processing_status.set("No video files found")
                self.log("No video files found in the selected folder")
                return
            
            total_files = len(video_files)
            self.log(f"Found {total_files} video files to process")
            
            # Process each video
            for i, video_file in enumerate(video_files):
                video_path = os.path.join(folder_path, video_file)
                self.current_file.set(video_file)
                self.progress_var.set((i / total_files) * 100)
                
                # Generate output filename (same as video but with .txt extension)
                output_file = os.path.splitext(video_path)[0] + ".txt"
                
                try:
                    self.process_video(video_path, output_file)
                except Exception as e:
                    self.log(f"Error processing {video_file}: {str(e)}")
            
            self.progress_var.set(100)
            self.processing_status.set("Processing complete!")
            self.log("All videos processed successfully")
            messagebox.showinfo("Success", "All videos have been processed!")
            
        except Exception as e:
            self.processing_status.set("Error")
            self.log(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def process_video(self, video_path, output_file):
        video_name = os.path.basename(video_path)
        self.log(f"Processing video: {video_name}")
        
        # Step 1: Transcribe with Whisper
        self.processing_status.set(f"Transcribing {video_name}...")
        self.log("Transcribing audio...")
        
        result = self.whisper_model.transcribe(video_path)
        transcription = result["text"]
        
        self.log(f"Transcription complete ({len(transcription)} characters)")
        
        # Step 2: Summarize with Claude
        self.processing_status.set(f"Summarizing {video_name}...")
        self.log("Generating summary with Claude...")
        
        summary = self.generate_summary(transcription, video_name)
        
        # Step 3: Save output
        self.log(f"Saving results to {os.path.basename(output_file)}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"TRANSCRIPTION AND SUMMARY OF: {video_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write("SUMMARY:\n")
            f.write("=" * 80 + "\n")
            f.write(summary)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("FULL TRANSCRIPTION:\n")
            f.write("=" * 80 + "\n")
            f.write(transcription)
        
        self.log(f"✓ Completed processing {video_name}")
    
    def generate_summary(self, transcription, video_name):
        try:
            # Truncate transcription if it's too long to save costs
            max_chars = 25000  # Claude's small model typically has a smaller context window
            if len(transcription) > max_chars:
                self.log(f"Transcription too long ({len(transcription)} chars), truncating to {max_chars} chars")
                transcription = transcription[:max_chars] + "...[truncated]"
            
            prompt = f"""Please provide a concise summary of the following video transcription. 
The summary should:
1. Identify the main topics and key points
2. Extract important insights or findings
3. Be clear and well-structured with bullet points for key information
4. Be around 300-500 words

Video title: {video_name}

TRANSCRIPTION:
{transcription}

Summary:"""

            self.log("Sending request to Claude...")
            response = self.anthropic_client.messages.create(
                model="claude-instant-1",  # Claude's small model
                max_tokens=1000,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.content[0].text
            self.log(f"Generated summary ({len(summary)} characters)")
            return summary
            
        except Exception as e:
            self.log(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

def main():
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()