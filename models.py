from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import bcrypt

db = SQLAlchemy()

class Voter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    face_encoding = db.Column(db.Text, nullable=True)
    is_verified = db.Column(db.Boolean, default=False)
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class OTPVerification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    otp = db.Column(db.String(6), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_used = db.Column(db.Boolean, default=False)
    temp_face_encoding = db.Column(db.Text, nullable=True)

#  Vote
class Vote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    voter_id = db.Column(db.Integer, db.ForeignKey('voter.id'), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.candidate_id'), nullable=False)
    election_id = db.Column(db.Integer, db.ForeignKey('election.election_id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    voter = db.relationship('Voter', backref='votes')
    candidate = db.relationship('Candidate', backref='votes')
    election = db.relationship('Election', backref='votes')
    __table_args__ = (
        db.UniqueConstraint('voter_id', 'election_id', name='unique_vote_per_election'),
    )


    
# ADMIN

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    face_encoding = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))


#    Candidate 

class Candidate(db.Model):
    candidate_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    election_id = db.Column(db.Integer, db.ForeignKey('election.election_id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    party = db.Column(db.String(100), nullable=False)
    election_title = db.Column(db.String(200), nullable=False)
    dob = db.Column(db.String(20), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    address = db.Column(db.Text, nullable=False)
    education = db.Column(db.String(100), nullable=False)
    symbol = db.Column(db.String(100), nullable=False)
    photo_filename = db.Column(db.String(200), nullable=True) 
    description = db.Column(db.Text, nullable=True)
    __table_args__ = (
        db.UniqueConstraint('election_id', 'email', name='unique_email_per_election'),  
    )
    

class Election(db.Model):
    election_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(100), nullable=False)
    level = db.Column(db.String(100), nullable=False)
    region = db.Column(db.String(200))
    nomination_start = db.Column(db.Date, nullable=False)
    nomination_end = db.Column(db.Date, nullable=False)
    voting_start = db.Column(db.DateTime, nullable=False)
    voting_end = db.Column(db.DateTime, nullable=False)
    result_date = db.Column(db.Date)
    voting_method = db.Column(db.String(100))
    max_votes = db.Column(db.Integer)
    seats = db.Column(db.Integer)
    face_required = db.Column(db.String(10))
    otp_required = db.Column(db.String(10))
    min_age = db.Column(db.Integer)
    id_required = db.Column(db.String(10))
    description = db.Column(db.Text)
    status = db.Column(db.String(20), default='Draft')
    is_assigned = db.Column(db.Boolean, default=True)
    show_live_results = db.Column(db.Boolean, default=False)
    candidates = db.relationship('Candidate', backref='election', lazy=True)

# Recent Activity
class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


    
