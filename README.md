# ✋ AirNav: Gesture-Based Browser Control with Gesture Password Unlock  

Control your browser using **hand gestures** powered by **OpenCV, MediaPipe, and Streamlit**.  
This project introduces a **Gesture Password Unlock System**, where the browser can only be controlled after performing a **custom sequence of hand gestures** (like a gesture-based PIN).  

---

## 🚀 Features  
- 🎥 **Hand Gesture Detection** (Move, Click, Scroll, Open Tabs)  
- 🔒 **Gesture Password Unlock** (sequence required before enabling control)  
- 🖥️ **Real-time Browser Control** (cursor, scroll, tab switching)  
- 🎛️ **Streamlit UI** (Start/Stop buttons, instructions, status display)  
- ✨ Professional design with side panel instructions  
- 👨‍💻 Made by **Onkar Virakt**  

---

## 🛠️ Tech Stack  
- **Python**  
- **OpenCV** – real-time video processing  
- **MediaPipe** – hand tracking  
- **PyAutoGUI** – simulating mouse/keyboard control  
- **Pynput** – advanced keyboard control  
- **Streamlit** – interactive UI  

---

## 🌍 Applications  
This project is not just experimental but has **real-world use cases**:  

1. 🧑‍🦽 **Assistive Technology** – enables physically disabled users to interact with computers without a mouse/keyboard.  
2. 🏥 **Touchless Control in Healthcare/Public Spaces** – hygienic browsing without direct contact, useful in hospitals or kiosks.  
3. 🚀 **Futuristic Workspaces** – intuitive interaction for AR/VR systems or gesture-based smart offices.  
4. 🕹️ **Gaming & Entertainment** – can be extended into gesture-based gaming controls.  
5. 🛡️ **Security Systems** – gesture password unlock can be adapted for secure access to applications or devices.  

---

## 📸 UI Preview  
Here’s how the app looks in action:  

![AirNav UI Preview](image/preview.png)  



---

## ▶️ How to Run  
```bash
git clone https://github.com/Onkar4578/AirNav.git
cd AirNav
pip install -r requirements.txt
streamlit run main.py

