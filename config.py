import os
from datetime import timedelta

class Config:
    SECRET_KEY = 'Topsecreate'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///voters.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Email Configuration
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'smartvoting.verify@gmail.com'
    MAIL_PASSWORD = 'tdgpjvdyclmiyzss'
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)
    
    # Upload Configuration
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
