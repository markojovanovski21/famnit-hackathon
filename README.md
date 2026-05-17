Graphs representing Vertical Wheel Acceleration(front and rear) can be found in Wheel_Acceleration/plotFunction+csv+img

# BumpIQ - Intelligent Speed Bump Analysis System

Developed by **Team Kernel Panic** for the **FAMNIT Hackathon 2026**.

## 📌 Project Overview
**BumpIQ** is an intelligent computer vision and motion analysis system designed to help autonomous and AI-assisted vehicles determine the optimal speed for crossing speed bumps. By leveraging real-time data processing, advanced object detection, and vehicle response logging, the system provides a robust solution for tracking micro-geometries and vehicle suspension behavior.

## 🛠️ The Challenge & The Solution
### The Problem
* **Barrel Distortion Effect:** Wide-angle lenses introduce optical distortion where horizontal lines curve away from the optical axis, leading to inaccurate bounding boxes.
* **Tracking Instability:** Standard vehicle-level bounding boxes are too unstable and unreliable for precise micro-geometry tracking and suspension analysis calculations.

### Our Innovation
* **Zebra-Striped Preprocessing & Correction:** Advanced video reshaping and resizing to comply with industry standards for real-time edge processing.
* **Dual-Bounding Box Strategy:** A custom architecture using a dual-tracking approach that logs both the overall vehicle body and individual wheels.
* **Micro-Displacement Tracking:** Measures the exact vertical distance variation (vibration effect in pixels) between the wheel well and the custom-detected wheel bounding box, removing the need for auxiliary physical markers.

## 📊 Experimental Results & Data Logging
BumpIQ was comprehensively evaluated across **3 different vehicle categories and suspension styles** (including a Skoda, a Passat, and a specialized EV platform). 
* **Computer Vision Metrics:** Achieved high-precision confidence scores (e.g., Car detection at `0.88`, Wheel detection at `0.91` - `0.92`).
* **Motion Profiling:** Extracts raw sensor data to log a complete *Vertical Wheel Oscillation Profile* and *Vertical Acceleration Signature* (measuring acceleration in $m/s^2$ over time) to map the static equilibrium baseline against real-time impact spikes.

## ☁️ Cloud Architecture & Tech Stack
BumpIQ leverages **Google Cloud Services** to support high-throughput, scalable, and AI-powered workflows:
* **Firebase:** Powering the responsive frontend web integration and secure user authentication.
* **Cloud Firestore:** Hosting analysis metrics, user sessions, and comprehensive processing histories.
* **Google Cloud Storage:** Scalable object storage holding raw video footage uploads and output processed media.
* **Gemini API:** Providing state-of-the-art AI-assisted evaluation and deep contextual analysis of vehicle response profiles.

---
*Created with 💻 by Team Kernel Panic.*
