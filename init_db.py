# init_db.py

# from app import app, db
# from models import Candidate  # Optional: also import Admin, Voter if you want all tables

# with app.app_context():
#     print("🛠️ Recreating all tables...")
#     db.create_all()

#     print("✅ Database reset successful.")


from models import Voter
import json
import numpy as np

v = Voter.query.filter_by(email="your_voter_email@example.com").first()
encoding = json.loads(v.face_encoding)
arr = np.array(encoding)

print("Type:", type(encoding))
print("Shape:", arr.shape)
print("Preview:", encoding[:5])
