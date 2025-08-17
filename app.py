from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_mail import Mail, Message
from flask_migrate import Migrate
from models import db,Vote, Voter, OTPVerification, Admin, Candidate, Election, ActivityLog
from config import Config
import random
import string
import cv2
import face_recognition
import numpy as np
import base64
import os
from datetime import datetime, timedelta
import json
import uuid
from werkzeug.utils import secure_filename
from fpdf import FPDF
import io
from flask import send_file
from flask import make_response
import csv
from sqlalchemy import func


app = Flask(__name__)
app.config.from_object(Config)


# Session configuration
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.permanent_session_lifetime = timedelta(minutes=30)
app.config['CANDIDATE_PHOTO_FOLDER'] = os.path.join('static', 'uploads', 'candidate_photos')
app.config['SYMBOL_FOLDER'] = os.path.join('static', 'uploads', 'symbols')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

db.init_app(app)
mail = Mail(app)
Migrate= Migrate(app, db)

# Create upload directory
UPLOAD_FOLDER = app.config.get('UPLOAD_FOLDER', 'uploads/')
os.makedirs(app.config['CANDIDATE_PHOTO_FOLDER'], exist_ok=True)
os.makedirs(app.config['SYMBOL_FOLDER'], exist_ok=True)

# Face scanning configuration
DATA_DIR = 'data/'
FRAMES_TOTAL = 51
CAPTURE_AFTER_FRAME = 3

# Initialize data directory
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize face detector
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Session storage for face scanning
user_face_sessions = {}

def log_activity(message):
    db.session.add(ActivityLog(message=message))
    db.session.commit()

@app.before_request
def create_tables_once():
    if not hasattr(app, '_tables_created'):
        with app.app_context():
            db.create_all()
            app._tables_created = True

@app.before_request
def make_session_permanent():
    session.permanent = True

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def send_otp_email(email, otp, purpose="registration"):
    try:
        if purpose == "login":
            subject = 'Voter Login OTP'
            title = 'Voter Login OTP'
        else:
            subject = 'Voter Registration OTP'
            title = 'Voter Registration OTP'
            
        msg = Message(
            subject,
            sender=app.config['MAIL_USERNAME'],
            recipients=[email]
        )
        msg.html = f'''
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #4CAF50;">{title}</h2>
            <p>Your OTP for voter {purpose} is:</p>
            <h1 style="color: #2196F3; font-size: 2em; letter-spacing: 5px;">{otp}</h1>
            <p>This OTP is valid for 10 minutes.</p>
        </div>
        '''
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

# Voter Registration Routes       <-------------------------------------------------------------->
@app.route('/register/step1', methods=['GET', 'POST'])
def register_step1():
    if request.method == 'POST':
        data = request.get_json()
        first_name = data.get('first_name')
        email = data.get('email')
        
        if not first_name or not email:
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        existing_voter = Voter.query.filter_by(email=email).first()
        if existing_voter:
            return jsonify({'success': False, 'message': 'This email is already registered'})
        
        otp = generate_otp()
        
        
        OTPVerification.query.filter_by(email=email).delete()
        db.session.commit()
        
        otp_record = OTPVerification(email=email, otp=otp)
        db.session.add(otp_record)
        db.session.commit()
        
        if send_otp_email(email, otp, "registration"):
            session['registration_data'] = {
                'first_name': first_name,
                'email': email,
                'step': 1,
                'otp_verified': False
            }
            session.modified = True
            return jsonify({'success': True, 'message': 'OTP sent to your email'})
        else:
            return jsonify({'success': False, 'message': 'Error sending email'})
    
    return render_template('register_step1.html')

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    otp = data.get('otp')
    
    if 'registration_data' not in session:
        return jsonify({'success': False, 'message': 'Session expired'})
    
    email = session['registration_data']['email']
    otp_record = OTPVerification.query.filter_by(
        email=email,
        otp=otp,
        is_used=False
    ).first()
    
    if not otp_record:
        return jsonify({'success': False, 'message': 'Invalid OTP'})
    
    if datetime.utcnow() - otp_record.created_at > timedelta(minutes=10):
        return jsonify({'success': False, 'message': 'OTP has expired'})
    
    otp_record.is_used = True
    db.session.commit()
    
    session['registration_data']['step'] = 2
    session['registration_data']['otp_verified'] = True
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': 'OTP verified successfully',
        'redirect_url': '/register/step2'
    })

@app.route('/register/step2')
def register_step2():
    if 'registration_data' not in session:
        return redirect(url_for('register_step1'))
    
    reg_data = session['registration_data']
    if not reg_data.get('otp_verified') or reg_data.get('step') != 2:
        return redirect(url_for('register_step1'))
    
    return render_template('register_step2.html', registration_data=reg_data)

@app.route('/api/start-face-scan', methods=['POST'])
def start_face_scan():
    data = request.get_json()
    user_id = data.get('email', 'temp_user')
    if not user_id:
        return jsonify({'error': 'Missing email'}), 400

    
    user_face_sessions[user_id] = {
        'faces_data': [],
        'frame_count': 0
    }
    
    return jsonify({
        'message': 'Face scanning started',
        'user_id': user_id,
        'total_frames_needed': FRAMES_TOTAL
    })

@app.route('/api/process-face-frame', methods=['POST'])
def process_face_frame():
    data = request.get_json()
    user_email = data.get('email')
    image_data = data.get('image')

    if not user_email or not image_data:
        return jsonify({'error': 'Missing email or image'}), 400

    user_id = user_email  

    if user_id not in user_face_sessions:
        user_face_sessions[user_id] = {
            'faces_data': [],
            'frame_count': 0,
            
        }
    face_session = user_face_sessions[user_id]

    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400

        #  Optimize: Convert to RGB and downscale for fast detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        face_locations_small = face_recognition.face_locations(small_frame, model='hog')
        face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations_small]

        face_detected = False
        if face_locations:
            face_session['frame_count'] += 1

            if (len(face_session['faces_data']) < FRAMES_TOTAL and
                face_session['frame_count'] % CAPTURE_AFTER_FRAME == 0):

                # Encode only when needed
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                if face_encodings:
                    face_session['faces_data'].append(face_encodings[0])
                    face_detected = True

        frames_captured = len(face_session['faces_data'])
        is_complete = frames_captured >= FRAMES_TOTAL

        response_data = {
            'face_detected': face_detected,
            'frames_captured': frames_captured,
            'total_frames_needed': FRAMES_TOTAL,
            'is_complete': is_complete,
            'progress_percentage': (frames_captured / FRAMES_TOTAL) * 100
        }

        if is_complete:
            success = save_face_data_to_db(user_id, face_session['faces_data'])
            face_session['face_saved'] = True 
            response_data['registration_success'] = success
            response_data['message'] = 'Face registration completed successfully' if success else 'Failed to save face data'
            response_data['step'] = session.get('registration_data', {}).get('step', None)

            #  Prevent repeat registration
            del user_face_sessions[user_id]
            response_data['completed'] = True

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Error processing frame: {str(e)}'}), 500



def save_face_data_to_db(user_id, faces_data):
    try:
        if not faces_data:
            print("❌ No face data provided.")
            return False

        # Average all face encodings
        faces_array = np.asarray(faces_data)
        avg_encoding = np.mean(faces_array, axis=0)

        print("Saving face encoding for email:", repr(user_id))

        otp_record = OTPVerification.query.filter_by(email=user_id)\
                    .order_by(OTPVerification.created_at.desc()).first()

        if otp_record:
            otp_record.temp_face_encoding = json.dumps(avg_encoding.tolist())
            db.session.commit()

            # ✅ Optional: mark registration step in session
            if 'registration_data' in session:
                session['registration_data']['step'] = 3
                session.modified = True
                print("✅ Step set to 3 in session:", session['registration_data'])
                
            else:
                print("⚠️ session['registration_data'] not found")

            print("✅ Face encoding saved temporarily in OTPVerification")
            return True
        else:
            print("❌ OTPVerification entry not found for:", repr(user_id))
            return False

    except Exception as e:
        print(f"❌ Error saving face encoding: {e}")
        return False


@app.route('/register/step3')
def register_step3():
    print("📍 Step3 Route Triggered")
    print("📦 session['registration_data']:", session.get('registration_data'))

    if 'registration_data' not in session:
        print("❌ No registration_data found in session.")
        return redirect(url_for('register_step1'))
    
    # reg_data = session['registration_data']
    # if reg_data.get('step') != 3:
    #     print(f"❌ Step not valid. Got step={reg_data.get('step')}, expected 3")
    #     return redirect(url_for('register_step1'))
    
    return render_template('register_step3.html')


@app.route('/complete-registration', methods=['POST'])
def complete_registration():
    try:
        data = request.get_json()
        password = data.get('password')
        confirm_password = data.get('confirm_password')

        if not password or not confirm_password:
            return jsonify({'success': False, 'message': 'All fields are required'})

        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'})

        if len(password) < 8:
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters'})

        if 'registration_data' not in session:
            return jsonify({'success': False, 'message': 'Session expired'})

        reg_data = session['registration_data']
        email = reg_data.get('email')

        #  Face data fetch from OTPVerification table
        otp_record = OTPVerification.query.filter_by(email=email).order_by(OTPVerification.created_at.desc()).first()
        if not otp_record or not otp_record.temp_face_encoding:
            return jsonify({'success': False, 'message': 'Face data missing. Please register again.'})

        voter = Voter(
            first_name=reg_data['first_name'],
            email=email,
            face_encoding=otp_record.temp_face_encoding,
            is_verified=True
        )
        voter.set_password(password)

        db.session.add(voter)
        db.session.commit()
        log_activity(f"New voter registered: {voter.first_name}")

        session.pop('registration_data', None)
        session.modified = True

        return jsonify({'success': True, 'message': 'Registration completed successfully'})

    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})




# Voter Login Routes                                   <....................................................................>
@app.route('/voter-login/step1', methods=['GET', 'POST'])
def voter_login_step1():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'})

        # Check if voter exists
        voter = Voter.query.filter_by(email=email).first()
        if not voter:
            return jsonify({'success': False, 'message': 'Email not registered. Please register first.'})

        # Check if voter is verified
        if not voter.is_verified:
            return jsonify({'success': False, 'message': 'Account not verified. Please complete registration.'})

        # Check password
        if not voter.check_password(password):
            return jsonify({'success': False, 'message': 'Incorrect password'})

        # Generate and send OTP
        otp = generate_otp()
        
        # Delete any existing OTP for this email
        OTPVerification.query.filter_by(email=email).delete()
        db.session.commit()
        
        otp_record = OTPVerification(email=email, otp=otp)
        db.session.add(otp_record)
        db.session.commit()

        if send_otp_email(email, otp, "login"):
            # Save voter login session
            session['voter_login'] = {
                'voter_id': voter.id,
                'email': voter.email,
                'first_name': voter.first_name,
                'step': 1,
                'otp_verified': False,
                'face_verified': False,
                'login_time': datetime.utcnow().isoformat()
            }
            session.modified = True
            return jsonify({'success': True, 'message': 'Credentials verified. OTP sent to your email.'})
        else:
            return jsonify({'success': False, 'message': 'Error sending OTP'})

    return render_template('voter_login_step1.html')

@app.route('/voter-login/step2')
def voter_login_step2():
    if 'voter_login' not in session or session['voter_login'].get('step') < 1:
        return redirect(url_for('voter_login_step1'))
    return render_template('voter_login_step2.html')

@app.route('/voter-login/verify-otp', methods=['POST'])
def voter_login_verify_otp():
    if 'voter_login' not in session or session['voter_login'].get('step') < 1:
        return jsonify({'success': False, 'message': 'Unauthorized access'})

    data = request.get_json()
    otp = data.get('otp')

    if not otp:
        return jsonify({'success': False, 'message': 'OTP is required'})

    email = session['voter_login']['email']
    otp_record = OTPVerification.query.filter_by(
        email=email,
        otp=otp,
        is_used=False
    ).first()

    if not otp_record:
        return jsonify({'success': False, 'message': 'Invalid OTP'})

    if datetime.utcnow() - otp_record.created_at > timedelta(minutes=10):
        return jsonify({'success': False, 'message': 'OTP has expired'})

    otp_record.is_used = True
    db.session.commit()

    session['voter_login']['step'] = 2
    session['voter_login']['otp_verified'] = True
    session.modified = True

    return jsonify({'success': True, 'message': 'OTP verified successfully'})

@app.route('/voter-login/resend-otp', methods=['POST'])
def voter_login_resend_otp():
    if 'voter_login' not in session:
        return jsonify({'success': False, 'message': 'Session expired'})

    email = session['voter_login']['email']
    otp = generate_otp()

    # Delete any existing OTP for this email
    OTPVerification.query.filter_by(email=email).delete()
    db.session.commit()

    otp_record = OTPVerification(email=email, otp=otp)
    db.session.add(otp_record)
    db.session.commit()

    if send_otp_email(email, otp, "login"):
        return jsonify({'success': True, 'message': 'OTP resent successfully'})
    else:
        return jsonify({'success': False, 'message': 'Error sending OTP'})

@app.route('/voter-login/step3')
def voter_login_step3():
    if ('voter_login' not in session or 
        session['voter_login'].get('step') < 2 or 
        not session['voter_login'].get('otp_verified')):
        return redirect(url_for('voter_login_step1'))
    
    return render_template('voter_login_step3.html')

@app.route('/voter-login/verify-face', methods=['POST'])
def voter_login_verify_face():
    if ('voter_login' not in session or 
        session['voter_login'].get('step') < 2 or 
        not session['voter_login'].get('otp_verified')):
        return jsonify({'success': False, 'message': 'Unauthorized access'})

    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({'success': False, 'message': 'Image data required'})

    try:
        voter_id = session['voter_login']['voter_id']
        voter = Voter.query.get(voter_id)
        
        if not voter:
            return jsonify({'success': False, 'message': 'Voter not found'})

        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'success': False, 'message': 'Could not decode image'})

        # Get stored face encoding
        stored_face_encoding = json.loads(voter.face_encoding)
        stored_face_array = np.array(stored_face_encoding)

        # Get face encoding from current image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return jsonify({'success': False, 'message': 'No face detected in image'})

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            return jsonify({'success': False, 'message': 'Could not encode face'})

        
        current_face_encoding = face_encodings[0]

        # Compare faces (using simple distance for now)
        # In production, use proper face recognition comparison
        face_distance = np.linalg.norm(stored_face_array - current_face_encoding)
        face_match = face_distance < 0.5  # Threshold for face matching
        
        if face_match:
            session['voter_login']['face_verified'] = True
            session['voter_login']['step'] = 3
            session['voter_id'] = voter.id
            session['voter_email'] = voter.email
            session.modified = True
            return jsonify({'success': True, 'message': 'Face verified successfully'})
        else:
            return jsonify({'success': False, 'message': 'Face verification failed. Please try again.'})

    except Exception as e:
        print(f"Face verification error: {e}")
        return jsonify({'success': False, 'message': 'Face verification error occurred'})

@app.route('/check-login-status')
def check_login_status():
    print("Check login status called")  # Debug log
    
    if ('voter_login' in session and 
        session['voter_login'].get('step') >= 3 and 
        session['voter_login'].get('face_verified')):
        
        print("Session found, fetching voter data")  # Debug log
        
        # Fetch fresh data from database
        voter_id = session['voter_login']['voter_id']
        voter = Voter.query.get(voter_id)
        
        if voter:
            print(f"Voter found: {voter.first_name}")  # Debug log
            return jsonify({
                'logged_in': True,
                'voter_name': voter.first_name,
                'email': voter.email,
                'voter_id': voter.id
            })
        else:
            print("Voter not found in database")  # Debug log
            return jsonify({'logged_in': False})
    else:
        print("No valid session found")  # Debug log
        return jsonify({'logged_in': False})

from sqlalchemy.orm import joinedload
import json
@app.route('/voter-dashboard')
def voter_dashboard():
    print("Dashboard route called")  # Debug log
    
    if ('voter_login' not in session or 
        session['voter_login'].get('step') < 3 or 
        not session['voter_login'].get('face_verified')):
        print("Redirecting to login - invalid session")  # Debug log
        return redirect(url_for('voter_login_step1'))

    voter_id = session['voter_login']['voter_id']
    voter = Voter.query.get(voter_id)
    
    if not voter:
        print("Voter not found, clearing session") 
        session.pop('voter_login', None)
        return redirect(url_for('voter_login_step1'))

    print(f"Rendering dashboard for voter: {voter.first_name}")  
    
    total_elections = Election.query.filter_by(is_assigned=True).count()
    live_elections = Election.query.options(joinedload(Election.candidates)).filter(
        Election.status.in_(['Live', 'Published']),
        Election.is_assigned == True
    ).all()
   
    published_elections = Election.query.filter_by(status='Published', is_assigned=True).all()
    live_election_ids = [e.election_id for e in live_elections]
    voted_election_ids = set(
        v.election_id for v in Vote.query.filter(
            Vote.voter_id == voter_id,
            Vote.election_id.in_(live_election_ids)
        ).all()
    )
      # ✅ Step: Set `e.voted = True` if already voted
    for election in live_elections:
        election.voted = (election.election_id in voted_election_ids)

    voted_count = len(voted_election_ids)
    
    # Prepare a JSON-serializable dict
    elections_dict = {
        e.election_id: {
            "title": e.title,
            "candidates": [
                {
                    "id": c.candidate_id,
                    "name": c.name,
                    "party": c.party,
                    "symbol": c.symbol
                } for c in e.candidates
            ]
        } for e in live_elections
    }


    # Update session with fresh data from database
    session['voter_login']['first_name'] = voter.first_name
    session.modified = True
    
    return render_template('voter_dashboard.html',
         voter=voter,
        elections=live_elections, voted_count=voted_count,published_elections=published_elections,

        total_elections=total_elections, elections_json=json.dumps(elections_dict) , now=datetime.now())

@app.route('/voter-profile')
def voter_profile():
    if ('voter_login' not in session or 
        session['voter_login'].get('step') < 3 or 
        not session['voter_login'].get('face_verified')):
        return redirect(url_for('voter_login_step1'))

    voter_id = session['voter_login']['voter_id']
    voter = Voter.query.get(voter_id)
    
    if not voter:
        session.pop('voter_login', None)
        return redirect(url_for('voter_login_step1'))

    return render_template('voter_profile.html', voter=voter)


@app.route('/logout')
def logout():
    session.pop('voter_login', None)
    session.pop('voter_id', None)
    session.pop('voter_email', None)
    session.pop('registration_data', None)
    return redirect(url_for('index'))

@app.route('/mark-face-verified', methods=['POST'])
def mark_face_verified():
    if 'voter_login' in session:
        session['voter_login']['face_verified'] = True
        session.modified = True
        return jsonify({'success': True})
    return jsonify({'success': False}), 401


# Vote casting with face verification

@app.route('/cast-vote', methods=['POST'])
def cast_vote():
    if ('voter_login' not in session or 
        session['voter_login'].get('step') < 3 or 
        not session['voter_login'].get('face_verified')):
        return jsonify({'success': False, 'message': 'Unauthorized access'})

    data = request.get_json()
    election_id = data.get('election_id')
    candidate_id = data.get('candidate_id')
    face_image = data.get('face_image')

    if not all([election_id, candidate_id, face_image]):
        return jsonify({'success': False, 'message': 'Missing required data'})

    try:
        voter_id = session['voter_login']['voter_id']
        voter = Voter.query.get(voter_id)

        if not voter:
            return jsonify({'success': False, 'message': 'Voter not found'})

        # Verify face again before voting
        if ',' in face_image:
            face_image = face_image.split(',')[1]

        image_bytes = base64.b64decode(face_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid face image'})

        # Get stored face encoding
        stored_face_encoding = json.loads(voter.face_encoding)
        stored_face_array = np.array(stored_face_encoding)

        # Get face encoding from current image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return jsonify({'success': False, 'message': 'No face detected'})

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            return jsonify({'success': False, 'message': 'Could not encode face'})

        current_face_encoding = face_encodings[0]

        # Compare faces
        distances = face_recognition.face_distance([stored_face_array], current_face_encoding)
        face_distance = distances[0]
        face_match = face_distance < 0.5

        print(f"Face Distance: {face_distance}, Match: {face_match}")

        if not face_match:
            return jsonify({'success': False, 'message': 'Face verification failed for voting'})

        # Check if vote already exists for this voter and election
        existing_vote = Vote.query.filter_by(
            voter_id=voter_id,
            election_id=election_id
        ).first()

        if existing_vote:
            return jsonify({'success': False, 'message': 'You have already voted in this election.'})

        
        # ✅ STORE THE VOTE
        new_vote = Vote(
            voter_id=voter_id,
            candidate_id=candidate_id,
            election_id=election_id
        )
        db.session.add(new_vote)
        db.session.commit()

        return jsonify({'success': True, 'message': '✅ Vote cast successfully!'})

    except Exception as e:
        print(f"Vote casting error: {e}")
        return jsonify({'success': False, 'message': 'Error casting vote'})

@app.route('/voter-results-page', methods=['GET', 'POST'])
def voter_results_page():
    if 'voter_id' not in session:
        flash("Please login to view results.", "error")
        return redirect('/voter-login')

    voter_id = session['voter_id']

    # ✅ Assigned Elections (better query)
    assigned_election_ids = db.session.query(Vote.election_id).filter_by(voter_id=voter_id).distinct().all()
    assigned_election_ids = [eid for (eid,) in assigned_election_ids]

    elections = Election.query.filter(Election.election_id.in_(assigned_election_ids)).all()
    elections = Election.query.filter(Election.status == 'Published').all()

    election = None
    candidates = []
    vote_counts = {}
    candidates_dict = {}
    winner_id = None
    candidate_names = []
    show_live = False

    if request.method == 'POST':
        election_id = request.form.get('election_id')
        election = Election.query.get(election_id)

        if not election:
            flash("Invalid election selected.", "error")
            return redirect('/voter-results-page')

        candidates = Candidate.query.filter_by(election_id=election_id).all()
        candidates_dict = {c.candidate_id: c for c in candidates}
        candidate_names = [c.name for c in candidates]

        # ✅ Vote count
        vote_data =db.session.query(
            Vote.candidate_id,
            func.count(Vote.id)
        ).filter(Vote.election_id == election_id).group_by(Vote.candidate_id).all()

        vote_counts = {cid: count for cid, count in vote_data}

        # ✅ Winner logic
        if vote_counts:
            winner_id = max(vote_counts, key=vote_counts.get)

        # ✅ Check if results are live
        show_live = election.show_live_results

    return render_template('voter_results_page.html',
        elections=elections,
        election=election,
        candidates=candidates,
        vote_counts=vote_counts,
        candidates_dict=candidates_dict,
        winner_id=winner_id,
        show_live=show_live,
        candidate_names=candidate_names
    )



@app.route('/voter_login')
def voter_login():
    return redirect(url_for('voter_login_step1'))

@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')

# Debug routes
@app.route('/debug-session')
def debug_session():
    return jsonify({
        'session_data': dict(session),
        'has_registration_data': 'registration_data' in session,
        'has_voter_login': 'voter_login' in session,
        'session_id': request.cookies.get('session')
    })

@app.route('/debug-voter-session')
def debug_voter_session():
    return jsonify({
        'voter_login_session': session.get('voter_login', {}),
        'session_id': request.cookies.get('session')
    })



# ADMIN            <.....................................................................................................>

@app.route('/admin-login')
def admin_login_redirect():
    return redirect(url_for('admin_login_step1')) 


      #step - 1

@app.route('/admin-login/step1', methods=['GET', 'POST'])
def admin_login_step1():
    if request.method == 'POST':
        print("✅ Received POST request")
        print("Headers:", request.headers)
        print("is_json:", request.is_json)
        print("Content-Type:", request.content_type)

        try:
            data = request.get_json()
            print("📦 Raw JSON data:", data)
        except Exception as e:
            print("❌ Error parsing JSON:", e)
            return jsonify({'success': False, 'message': 'Invalid JSON format'})

        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'})

        admin = Admin.query.filter_by(email=email).first()
        if not admin:
            return jsonify({'success': False, 'message': 'Admin not found'})

        if not admin.check_password(password):
            return jsonify({'success': False, 'message': 'Wrong password'})

        # 🔐 OTP Logic as before...
        otp = generate_otp()

        OTPVerification.query.filter_by(email=email).delete()
        db.session.commit()

        otp_record = OTPVerification(email=email, otp=otp)
        db.session.add(otp_record)
        db.session.commit()

        if send_otp_email(email, otp, "admin login"):
            session['admin_login'] = {
                'admin_id': admin.id,
                'email': admin.email,
                'step': 1,
                'otp_verified': False,
                'face_verified': False
            }
            session.modified = True
            return jsonify({'success': True, 'message': 'OTP sent successfully'})
        else:
            return jsonify({'success': False, 'message': 'Error sending OTP'})

    return render_template('admin_login_step1.html')




#      Step - 2
@app.route('/admin-login/step2', methods=['GET'])
def admin_login_step2():
    return render_template('admin_login_step2.html')

## OTP ko verify krenge  
@app.route('/admin-login/verify-otp', methods=['POST'])
def admin_login_verify_otp():
    if 'admin_login' not in session or session['admin_login'].get('step') < 1:
        return jsonify({'success': False, 'message': 'Session expired or unauthorized'})

    data = request.get_json()
    otp = data.get('otp')

    email = session['admin_login']['email']
    otp_record = OTPVerification.query.filter_by(email=email, otp=otp, is_used=False).first()

    if not otp_record:
        return jsonify({'success': False, 'message': 'Invalid OTP'})

    if datetime.utcnow() - otp_record.created_at > timedelta(minutes=10):
        return jsonify({'success': False, 'message': 'OTP expired'})

    otp_record.is_used = True
    db.session.commit()

    session['admin_login']['step'] = 2
    session['admin_login']['otp_verified'] = True
    session.modified = True

    return jsonify({'success': True, 'message': 'OTP verified successfully'})


             # step - 3

@app.route('/admin-login/face-verify', methods=['GET', 'POST'] )
def admin_face_verification():
    if 'admin_login' not in session or not session['admin_login'].get('otp_verified'):
        return redirect(url_for('admin_login_step1'))

    return render_template('admin_face_verification.html')


@app.route('/admin-login/verify-face', methods=[ 'POST'] )
def admin_verify_face():
    if 'admin_login' not in session or not session['admin_login'].get('otp_verified'):
        return jsonify({'success': False, 'message': 'Unauthorized access'})

    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({'success': False, 'message': 'Image required'})

    try:
        admin = db.session.get(Admin,session['admin_login']['admin_id'])

        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        stored_encoding = np.array(json.loads(admin.face_encoding))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_encodings:
            return jsonify({'success': False, 'message': 'No face detected'})

        current_encoding = face_encodings[0]
        face_distance = np.linalg.norm(stored_encoding - current_encoding)
        match = face_distance < 0.6

        if match:
            session['admin_login']['face_verified'] = True
            session['admin_login']['step'] = 3
            session['admin_logged_in'] = True
            session.modified = True
            return jsonify({'success': True, 'message': 'Face verified successfully'})
        else:
            return jsonify({'success': False, 'message': 'Face verification failed'})

    except Exception as e:
        print(f"Admin face verification error: {e}")
        return jsonify({'success': False, 'message': 'Error verifying face'})

# step - 3 wala template render krenge
@app.route('/admin-login/step3', methods=['GET'])
def admin_login_step3():
    if ('admin_login' not in session or 
        session['admin_login'].get('step') < 3 or 
        not session['admin_login'].get('face_verified')):
        return redirect(url_for('admin_login_step1'))

    return render_template('admin_login_step3.html')  


# #  Admin Face Registration
@app.route('/admin-register-face',methods=['GET'] )
def admin_register_face_page():
    return render_template('admin_register_face.html')

@app.route('/admin-register/face', methods=['POST'])
def admin_register_face():
    if 'admin_login' not in session or not session['admin_login'].get('otp_verified'):
        return jsonify({'success': False, 'message': 'Unauthorized'})

    data = request.get_json()
    image_data = data.get('image')
    if not image_data:
        return jsonify({'success': False, 'message': 'Image required'})

    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_img)
        if not locations:
            return jsonify({'success': False, 'message': 'No face detected'})

        encodings = face_recognition.face_encodings(rgb_img, locations)
        if not encodings:
            return jsonify({'success': False, 'message': 'Encoding failed'})

        admin = db.session.get(Admin,session['admin_login']['admin_id'])
        admin.face_encoding = json.dumps(encodings[0].tolist())
        db.session.commit()

        return jsonify({'success': True, 'message': 'Face registered successfully'})

    except Exception as e:
        print("Face registration error:", e)
        return jsonify({'success': False, 'message': 'Internal error'})


# Dashboard route for admin

@app.route('/admin-dashboard')
def admin_dashboard():
    if ('admin_login' not in session or 
        session['admin_login'].get('step') < 3 or 
        not session['admin_login'].get('face_verified')):
        return redirect(url_for('admin_login_step1'))
    elections = Election.query.all()
    if not elections:
        flash("No elections found.", "warning")
    admin = db.session.get(Admin,session['admin_login']['admin_id'])

    recent_activities = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(5).all()
    return render_template('admin_dashboard.html', admin=admin,elections=elections,recent_activities=recent_activities)

#ADMIN Logout/Login
@app.route('/admin-logout')
def admin_logout():
    session.pop('admin_login', None)
    return redirect(url_for('index'))


# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


# Manage Admin Dashboard   <-------------------------------------------------> 


@app.route('/view-voters')
def view_voters():
    voters = Voter.query.all()
    return render_template('view_voters.html', voters=voters)

@app.route('/add-voter', methods=['GET', 'POST'])
def add_voter():
    if request.method == 'POST':
        first_name = request.form['first_name']
        email = request.form['email']
        password = request.form['password']
        new_voter = Voter(first_name=first_name, email=email, password_hash=password)
        db.session.add(new_voter)
        db.session.commit()
        log_activity(f"New voter registered: {first_name}")
        return redirect('/view-voters')
    return render_template('add_voter.html')

@app.route('/delete-voter', methods=['GET', 'POST'])
def delete_voter():
    if request.method == 'POST':
        email = request.form['email']
        voter = Voter.query.filter_by(email=email).first()
        if voter:
            db.session.delete(voter)
            db.session.commit()
        return redirect('/view-voters')
    return render_template('delete_voter.html')

@app.route('/export-voter-list')
def export_list():
    return render_template('export_list.html')

@app.route('/export-voter-list/download')
def download_voter_list():
    voters = Voter.query.all()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Name', 'Email'])
    for v in voters:
        cw.writerow([v.id, v.first_name, v.email])
    output = io.BytesIO()
    output.write(si.getvalue().encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype='text/csv', download_name='voter_list.csv', as_attachment=True)


#    Manage Candidates Dashboard   <------------------------------------------------->

# Add Candidate
import os
from werkzeug.utils import secure_filename
from models import db, Candidate
from flask import  flash
from werkzeug.exceptions import RequestEntityTooLarge
import traceback
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit


@app.route('/add-candidate', methods=['GET', 'POST'])
def add_candidate():
    if request.method == 'POST':
        try:
            name = request.form['name']
            party = request.form['party']
            dob = request.form['dob']
            gender = request.form['gender']
            email = request.form['email']
            phone = request.form['phone']
            address = request.form['address']
            education = request.form['education']
            description = request.form['description']
            election_id = request.form['election_id']
            election= Election.query.get(election_id)

            # Handle photo upload
            photo = request.files['photo']
            photo_filename = None
            if photo and photo.filename:
                photo_filename = secure_filename(photo.filename)
                photo.save(os.path.join(app.config['CANDIDATE_PHOTO_FOLDER'], photo_filename))
                

            else:
                flash("⚠️ Photo is required.", "warning")
                return redirect(url_for('add_candidate'))
            # 🔥 Handle symbol upload 
            symbol_file = request.files['symbol']
            symbol_filename = None
            if symbol_file and symbol_file.filename:
                symbol_filename = secure_filename(symbol_file.filename)
                symbol_file.save(os.path.join(app.config['SYMBOL_FOLDER'], symbol_filename))
            else:
                flash("⚠️ Symbol is required.", "warning")
                return redirect(url_for('add_candidate')) 

            # Save to DB
            new_candidate = Candidate(
                election_id=election_id,
                name=name,
                party=party,
                election_title=election.title,
                dob=dob,
                gender=gender,
                email=email,
                phone=phone,
                address=address,
                education=education,
                symbol=symbol_filename,
                photo_filename=photo_filename,
                description=description
               

            )
            db.session.add(new_candidate)
            db.session.commit()
            log_activity(f"Candidate '{name}' added for election '{election.title}'")
            candidate_id = new_candidate.candidate_id
            # flash("Candidate added successfully!", "success")
            return redirect(url_for('add_candidate', success=1,cid=candidate_id))
        

        except Exception as e:
            traceback.print_exc()
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for('add_candidate'))
        
    success = request.args.get('success')
    candidate_id = request.args.get('cid')
    elections = Election.query.filter_by(status='Live').all()
    return render_template("add_candidate.html", success=success,candidate_id=candidate_id, elections=elections)

@app.errorhandler(413)
def too_large(e):
    flash("⚠️ File too large! Maximum allowed size is 16MB.", "warning")
    return redirect(url_for('add_candidate'))


# Candidate List
@app.route('/candidate-info', methods=['GET', 'POST'])
def candidate_info():
    elections = Election.query.all()
    selected_election = None
    candidates = []

    if request.method == 'POST':
        election_id = request.form.get('election_id')
        if election_id:
            selected_election = Election.query.get(election_id)
            candidates = Candidate.query.filter_by(election_id=election_id).all()

        else:
            candidates =[]
    else:
       candidates = []

    return render_template(
        'candidate_info.html',
        elections=elections,
        candidates=candidates,
        show_delete=False,
        election=selected_election
    )


@app.route('/remove-candidate')
def remove_candidate():
    candidates = Candidate.query.all()
    return render_template("candidate_info.html", candidates=candidates, show_delete=True)
@app.route('/delete-candidate/<int:candidate_id>', methods=['POST'])
def delete_candidate(candidate_id):
    candidate = Candidate.query.get_or_404(candidate_id)
    db.session.delete(candidate)
    db.session.commit()
    flash("Candidate deleted successfully!", "success")
    return redirect(url_for('remove_candidate'))


# Create Election
from models import Election
@app.route('/create-election', methods=['GET', 'POST'])
def create_election():
    if request.method == 'POST':
        try:
            election = Election(
                title=request.form['title'],
                type=request.form['type'],
                level=request.form['level'],
                region=request.form['region'],
                nomination_start=datetime.strptime(request.form['nomination_start'], '%Y-%m-%d'),
                nomination_end=datetime.strptime(request.form['nomination_end'], '%Y-%m-%d'),
                voting_start=datetime.strptime(request.form['voting_start'], '%Y-%m-%dT%H:%M'),
                voting_end=datetime.strptime(request.form['voting_end'], '%Y-%m-%dT%H:%M'),
                result_date=datetime.strptime(request.form['result_date'], '%Y-%m-%d'),
                voting_method=request.form['voting_method'],
                max_votes=int(request.form['max_votes']),
                seats=int(request.form['seats']),
                face_required=request.form['face_required'],
                otp_required=request.form['otp_required'],
                min_age=int(request.form['min_age']),
                id_required=request.form['id_required'],
                description=request.form['description'],
                status='Draft' 
            )
            db.session.add(election)
            db.session.commit()
            log_activity(f"Election '{election.title}' created")
            flash("Election created successfully!", "success")
            return redirect(url_for('assigned_election'))
        except Exception as e:
            db.session.rollback()
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for('create_election'))

    return render_template('create_election.html')

#    Assigned Election
@app.route('/assigned-election')
def assigned_election():
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login_step1'))

    elections = Election.query.filter(
        Election.status.in_(['Draft','Live','Published']),
        # Election.is_assigned==True
    ).all()
    return render_template('assigned_election.html', elections=elections)

@app.route('/assign-election/<int:election_id>', methods=['POST'])
def assign_election(election_id):
    if 'admin_logged_in' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized access'}), 401

    try:
        election = Election.query.get(election_id)
        if election:
            election.is_assigned = True
            db.session.commit()
            return jsonify({'success': True, 'message': 'Election assigned successfully'})
        else:
            return jsonify({'success': False, 'message': 'Election not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500



@app.route('/unassign-election/<int:election_id>', methods=['POST'])
def unassign_election(election_id):
    try:
        election = Election.query.get(election_id)
        if election:
            election.is_assigned = False  
            db.session.commit()
            return jsonify({'success': True, 'message': 'Election unassigned successfully'})
        else:
            return jsonify({'success': False, 'message': 'Election not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/make-election-live/<int:election_id>', methods=['POST'])
def make_election_live(election_id):
    election = Election.query.get(election_id)
    if election:
        election.status = 'Live'
        election.is_assigned = True 
        db.session.commit()
    return redirect(url_for('assigned_election'))

@app.route('/end-election/<int:election_id>', methods=['POST'])
def end_election(election_id):
    election = Election.query.get(election_id)
    if election:
        election.status = 'Ended'
        db.session.commit()
        log_activity(f"Election '{election.title}' marked as Ended")
    return redirect(url_for('assigned_election'))

# Sedule Voting
@app.route('/schedule-voting', methods=['POST'])
def schedule_voting():
    data = request.get_json()
    election_id = data.get('election_id')
    type = data.get('type')
    party = data.get('party')
    voting_start = data.get('voting_start')
    voting_end = data.get('voting_end')

    try:
        election = Election.query.get(election_id)
        election.voting_start = datetime.fromisoformat(voting_start)
        election.voting_end = datetime.fromisoformat(voting_end)
        election.is_assigned = True
        db.session.commit()
        return jsonify({'message': 'Voting schedule updated successfully!'})
    except Exception as e:
        print("Schedule error:", e)
        return jsonify({'message': 'Failed to update voting schedule'}), 500


# MAnage Election Results
@app.route('/admin-results-page', methods=['GET','POST'])
def admin_results_page():
    elections = Election.query.all()

    selected_election = None
    candidates = []
    candidates_dict = {}
    vote_counts = {}
    winner_id = None

    if request.method == 'POST':
        election_id = request.form.get('election_id')
        if election_id:
            selected_election = Election.query.get(election_id)
            session['current_election_id'] = election_id
            candidates = Candidate.query.filter_by(election_id=election_id).all()
            candidates_dict = {candidate.candidate_id: candidate for candidate in candidates}
            vote_counts = {
                c.candidate_id: Vote.query.filter_by(candidate_id=c.candidate_id).count()
                for c in candidates
                }
            winner_id = max(vote_counts, key=vote_counts.get) if vote_counts else None

    return render_template('admin_results_page.html',
    elections=elections,
    election=selected_election,
    selected_election=selected_election,
    candidates=candidates,
    candidates_dict=candidates_dict,
    vote_counts=vote_counts if vote_counts else {},  
    winner_id=winner_id,
    candidate_names=[c.name for c in candidates] if candidates else [], 
    role='admin',
    show_live=True if selected_election and selected_election.status == 'Live' else False

    )

@app.route('/admin/toggle-live', methods=['POST'])
def toggle_live():
    election_id = session.get('current_election_id')
    if not election_id:
        return jsonify({'message': 'Election not selected'}), 400

    election = Election.query.get(election_id)
    if not election:
        return jsonify({'message': 'Election not found'}), 404

    election.show_live_results = not election.show_live_results
    db.session.commit()
    return jsonify({'message': f'Live results {"enabled" if election.show_live_results else "disabled"}'})

@app.route('/admin/publish-results', methods=['POST'])
def publish_results():
    election_id = session.get('current_election_id')
    if not election_id:
        return jsonify({'message': 'Election not selected'}), 400

    election = Election.query.get(election_id)
    if not election:
        return jsonify({'message': 'Election not found'}), 404

    election.status = 'Published'
    db.session.commit()
    return jsonify({'message': 'Results published successfully'})



@app.route('/admin/generate-report')
def generate_report():
    from fpdf import FPDF

    results = db.session.query(Candidate.name, Candidate.party, db.func.count(Vote.voter_id))\
        .outerjoin(Vote, Candidate.candidate_id == Vote.candidate_id)\
        .group_by(Candidate.candidate_id).all()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Election Results Report", ln=True, align='C')

    for name, party, count in results:
        pdf.cell(200, 10, txt=f"{name} ({party}) - {count} votes", ln=True)

    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=election_report.pdf'
    return response

@app.route('/admin/export-votes')
def export_votes():
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Voter ID', 'Candidate ID', 'Election ID'])

    votes = Vote.query.all()
    for v in votes:
        writer.writerow([v.voter_id, v.candidate_id, v.election_id])

    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=voting_data.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        from models import Admin
        import bcrypt
        
        password = 'admin123'  
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        existing_admin = Admin.query.filter_by(email='sukesh2294@gmail.com').first()
        if not existing_admin:
            admin = Admin( 
                full_name='Sukesh Kumar',
                email='sukesh2294@gmail.com',
                password_hash=hashed,
                created_at=datetime.utcnow()
                )
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin added.")
        else:
            print("⚠️ Admin already exists.")
    

app.run(debug=True)
