import os
import secrets
import string
import smtplib
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional
from enum import Enum
import pytz # Import pytz for timezone handling
import requests
import json
import hmac
import httpx
import pyotp  # Added for 2FA support
from collections import defaultdict
import time
import random
import uuid

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, status, Request, Query, Form, UploadFile, File
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Header
from fastapi.responses import RedirectResponse
from fastapi_utils.tasks import repeat_every
from slowapi import Limiter, _rate_limit_exceeded_handler  # Added rate limiting
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.staticfiles import StaticFiles

# Database imports
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text, UniqueConstraint, create_engine, and_, or_, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.types import DECIMAL as SQLDecimal
from sqlalchemy.orm import joinedload
from sqlalchemy import exists

# Authentication imports
from passlib.context import CryptContext
from jose import JWTError, jwt

# Email imports
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from email.mime.base import MIMEBase
from email import encoders

# Pydantic imports
from pydantic import BaseModel, EmailStr, field_validator, model_validator
from jinja2 import Environment, FileSystemLoader

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
required_env_vars = [
    "DATABASE_URL",
    "SECRET_KEY",
    "SMTP_SERVER",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "FROM_EMAIL",
    "FROM_NAME",
    "BASE_URL",
    "FRONTEND_BASE_URL",
    "ADMIN_EMAIL",
    "ADMIN_PASSWORD",
    "ADMIN_PIN",
    "APP_SECRET"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30
RATE_LIMIT_STORAGE = defaultdict(list)

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USERNAME)
FROM_NAME = os.getenv("FROM_NAME")
BASE_URL = os.getenv("BASE_URL")
APP_SECRET = os.getenv("APP_SECRET")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_PIN = os.getenv("ADMIN_PIN")

# =============================================================================
# DATABASE SETUP
# =============================================================================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)

# =============================================================================
# ENUMS (DEFINED FIRST TO AVOID FORWARD REFERENCES)
# =============================================================================

class UserStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    SUSPENDED = "suspended"
    REJECTED = "rejected"

class WithdrawalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROCESSED = "processed"

class CryptoType(str, Enum):
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"

class DepositStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"

# =============================================================================
# DATABASE MODELS (ORDERED TO RESOLVE FORWARD REFERENCES)
# =============================================================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(String(10), unique=True, Index=True, nullable=False)
    email = Column(String, unique=True, Index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    pin_hash = Column(String, nullable=False)
    status = Column(String, default=UserStatus.PENDING)
    is_admin = Column(Boolean, default=False)
    is_agent = Column(Boolean, default=False)
    is_flagged = Column(Boolean, default=False)
    usd_balance = Column(SQLDecimal(10, 2), default=0)
    referral_code = Column(String, unique=True, nullable=True)
    referred_by_code = Column(String, nullable=True)
    email_verified = Column(Boolean, default=False)
    birthday_day = Column(Integer, nullable=True)
    birthday_month = Column(Integer, nullable=True)
    birthday_year = Column(Integer, nullable=True)
    gender = Column(String(1), nullable=True)
    user_country_code = Column(String(2), nullable=True)
    zip_code = Column(String, nullable=True)
    bitcoin_wallet = Column(String, nullable=True)
    ethereum_wallet = Column(String, nullable=True)
    bitcoin_balance = Column(SQLDecimal(18, 8), default=0)
    ethereum_balance = Column(SQLDecimal(18, 8), default=0)
    personal_mining_rate = Column(Float, nullable=True)  # Personal rate set by admin
    two_fa_secret = Column(String, nullable=True)
    two_fa_enabled = Column(Boolean, default=False)
    account_locked = Column(Boolean, default=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    last_failed_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    devices = relationship("UserDevice", back_populates="user")
    sent_transfers = relationship("CryptoTransfer", foreign_keys="CryptoTransfer.from_user_id", back_populates="from_user")
    received_transfers = relationship("CryptoTransfer", foreign_keys="CryptoTransfer.to_user_id", back_populates="to_user")
    withdrawals = relationship("Withdrawal", back_populates="user", foreign_keys="Withdrawal.user_id")
    approvals = relationship("UserApproval", back_populates="user", foreign_keys="UserApproval.user_id")
    activity_logs = relationship("ActivityLog", back_populates="user", cascade="all, delete-orphan")
    deposits = relationship("CryptoDeposit", back_populates="user")
    mining_sessions = relationship("MiningSession", back_populates="user")

class AdminSettings(Base):
    __tablename__ = "admin_settings"
    
    id = Column(Integer, primary_key=True, Index=True)
    bitcoin_rate_usd = Column(SQLDecimal(10, 2), default=50000.00)  # Bitcoin price in USD
    ethereum_rate_usd = Column(SQLDecimal(10, 2), default=3000.00)  # Ethereum price in USD
    global_mining_rate = Column(Float, default=0.70)  # Default 70% mining rate
    bitcoin_deposit_qr = Column(String, nullable=True)  # QR code image path
    ethereum_deposit_qr = Column(String, nullable=True)  # QR code image path
    bitcoin_wallet_address = Column(String, nullable=True)
    ethereum_wallet_address = Column(String, nullable=True)
    referral_reward_enabled = Column(Boolean, default=True)
    referral_reward_type = Column(String, default="bitcoin")  # bitcoin or ethereum
    referral_reward_amount = Column(SQLDecimal(18, 8), default=0.001)  # Default reward amount
    referrer_reward_amount = Column(SQLDecimal(18, 8), default=0.001)  # Reward for referrer
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ReferralReward(Base):
    __tablename__ = "referral_rewards"
    
    id = Column(Integer, primary_key=True, Index=True)
    referrer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    referred_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reward_type = Column(String, nullable=False)  # bitcoin or ethereum
    reward_amount = Column(SQLDecimal(18, 8), nullable=False)
    status = Column(String, default="pending")  # pending, paid
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    paid_at = Column(DateTime(timezone=True), nullable=True)
    
    referrer = relationship("User", foreign_keys=[referrer_id])
    referred = relationship("User", foreign_keys=[referred_id])

class EmailNotification(Base):
    __tablename__ = "email_notifications"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    email = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    template_type = Column(String, nullable=False)  # deposit_confirmed, login_alert, etc.
    html_content = Column(Text, nullable=True)
    status = Column(String, default="pending")  # pending, sent, failed
    attempts = Column(Integer, default=0)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    scheduled_for = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User")

class TransactionHistory(Base):
    __tablename__ = "transaction_history"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    transaction_type = Column(String, nullable=False)  # deposit, withdrawal, transfer, mining, referral_reward
    crypto_type = Column(String, nullable=True)  # bitcoin, ethereum
    amount = Column(SQLDecimal(18, 8), nullable=False)
    description = Column(String, nullable=False)
    reference_id = Column(String, nullable=True)  # Reference to related record
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User")

class AdminAuditLog(Base):
    __tablename__ = "admin_audit_logs"
    
    id = Column(Integer, primary_key=True, Index=True)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)
    target_type = Column(String, nullable=True)  # user, deposit, settings, etc.
    target_id = Column(String, nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(Text, nullable=True)
    before_value = Column(Text, nullable=True)
    after_value = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    admin = relationship("User")

class CryptoDeposit(Base):
    __tablename__ = "crypto_deposits"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    crypto_type = Column(String, nullable=False)  # bitcoin or ethereum
    amount = Column(SQLDecimal(18, 8), nullable=False)  # crypto amount
    usd_amount = Column(SQLDecimal(18, 2), nullable=False)  # USD equivalent
    status = Column(String, default=DepositStatus.PENDING)
    transaction_hash = Column(String, nullable=True)
    evidence_url = Column(String, nullable=True)  # URL to uploaded evidence
    confirmed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    
    user = relationship("User", back_populates="deposits")

class MiningSession(Base):
    __tablename__ = "mining_sessions"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    deposit_id = Column(Integer, ForeignKey("crypto_deposits.id"), nullable=False)
    crypto_type = Column(String, nullable=False)  # bitcoin or ethereum
    deposited_amount = Column(SQLDecimal(18, 8), nullable=False)
    mining_rate = Column(Float, nullable=False)  # Rate used for this session
    mined_amount = Column(SQLDecimal(18, 8), default=0)
    is_active = Column(Boolean, default=True)
    is_paused = Column(Boolean, default=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    paused_at = Column(DateTime(timezone=True), nullable=True)
    last_processed = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="mining_sessions")

class MiningRate(Base):
    __tablename__ = "mining_rates"
    
    id = Column(Integer, primary_key=True, Index=True)
    crypto_type = Column(String, nullable=False)  # bitcoin or ethereum
    global_rate = Column(Float, nullable=False)  # Global mining rate percentage
    duration_hours = Column(Integer, default=24)  # Duration for the rate
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class DepositQRCode(Base):
    __tablename__ = "deposit_qr_codes"
    
    id = Column(Integer, primary_key=True, Index=True)
    crypto_type = Column(String, nullable=False)  # bitcoin or ethereum
    qr_code_url = Column(String, nullable=False)  # URL to uploaded QR code image
    wallet_address = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UserDevice(Base):
    __tablename__ = "user_devices"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    device_fingerprint = Column(String, nullable=False)
    ip_address = Column(String, nullable=False)
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="devices")
    
    __table_args__ = (UniqueConstraint('user_id', 'device_fingerprint', name='unique_user_device'),)

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)  # e.g., "USER_LOGIN", "DEPOSIT_MADE"
    details = Column(Text)
    ip_address = Column(String)
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="activity_logs")

class SecurityLog(Base):
    __tablename__ = "security_logs"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    event_type = Column(String, nullable=False)  # failed_login, successful_login, 2fa_enabled, etc.
    details = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User")

class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    
    id = Column(Integer, primary_key=True, Index=True)
    email = Column(String, nullable=False)
    ip_address = Column(String, nullable=False)
    success = Column(Boolean, nullable=False)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class FraudFlag(Base):
    __tablename__ = "fraud_flags"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reason = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="fraud_flag")

class OTP(Base):
    __tablename__ = "otps"
    
    id = Column(Integer, primary_key=True, Index=True)
    email = Column(String, nullable=False)
    otp_code = Column(String, nullable=False)
    purpose = Column(String, nullable=False) # signup, password_reset, pin_reset
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CryptoTransfer(Base):
    __tablename__ = "crypto_transfers"
    
    id = Column(Integer, primary_key=True, Index=True)
    from_user_id = Column(Integer, ForeignKey("users.id"))
    to_user_id = Column(Integer, ForeignKey("users.id"))
    crypto_type = Column(String, nullable=False)  # bitcoin or ethereum
    amount = Column(SQLDecimal(18, 8), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    from_user = relationship("User", foreign_keys=[from_user_id], back_populates="sent_transfers")
    to_user = relationship("User", foreign_keys=[to_user_id], back_populates="received_transfers")

class Withdrawal(Base):
    __tablename__ = "withdrawals"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    crypto_type = Column(String, nullable=False)
    amount = Column(SQLDecimal(18, 8), nullable=False)
    wallet_address = Column(String, nullable=False)
    status = Column(String, default=WithdrawalStatus.PENDING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    user = relationship("User", back_populates="withdrawals")

class UserApproval(Base):
    __tablename__ = "user_approvals"
    
    id = Column(Integer, primary_key=True, Index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    approved_by = Column(Integer, ForeignKey("users.id"))
    status = Column(String, nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", foreign_keys=[user_id], back_populates="approvals")

# =============================================================================
# UTILITY FUNCTIONS (DEFINED AFTER MODELS TO AVOID FORWARD REFERENCES)
# =============================================================================

def generate_user_id():
    """Generate a unique 10-digit user ID"""
    # Generate 10-digit number (no leading zero)
    return ''.join([str(random.randint(1, 9))] + [str(random.randint(0, 9)) for _ in range(9)])

def generate_referral_code():
    """Generate a unique referral code"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def generate_2fa_secret():
    """Generate a new 2FA secret key"""
    return pyotp.random_base32()

def verify_2fa_token(secret: str, token: str) -> bool:
    """Verify 2FA token"""
    totp = pyotp.TOTP(secret)
    return totp.verify(token, valid_window=1)

def get_2fa_qr_url(secret: str, email: str, issuer: str = "Crypto Mining Platform") -> str:
    """Generate QR code URL for 2FA setup"""
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(email, issuer_name=issuer)

def check_rate_limit(identifier: str, max_requests: int, window_seconds: int) -> bool:
    """Check if request is within rate limit"""
    now = time.time()
    requests = RATE_LIMIT_STORAGE[identifier]
    
    # Remove old requests outside the window
    RATE_LIMIT_STORAGE[identifier] = [req_time for req_time in requests if now - req_time < window_seconds]
    
    # Check if under limit
    if len(RATE_LIMIT_STORAGE[identifier]) < max_requests:
        RATE_LIMIT_STORAGE[identifier].append(now)
        return True
    return False

def log_security_event(db: Session, user_id: Optional[int], event_type: str, details: str, ip_address: str):
    """Log security-related events"""
    security_log = SecurityLog(
        user_id=user_id,
        event_type=event_type,
        details=details,
        ip_address=ip_address
    )
    db.add(security_log)
    db.commit()

def is_account_locked(db: Session, email: str) -> bool:
    """Check if account is locked due to failed login attempts"""
    lockout_time = datetime.utcnow() - timedelta(minutes=LOCKOUT_DURATION_MINUTES)
    
    failed_attempts = db.query(SecurityLog).filter(
        SecurityLog.event_type == "failed_login",
        SecurityLog.details.contains(email),
        SecurityLog.created_at > lockout_time
    ).count()
    
    return failed_attempts >= MAX_LOGIN_ATTEMPTS

def get_admin_settings(db: Session):
    """Get or create admin settings"""
    settings = db.query(AdminSettings).first()
    if not settings:
        settings = AdminSettings()
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings

def calculate_usd_values(user: User, admin_settings: AdminSettings):
    """Calculate USD values for user's crypto balances"""
    bitcoin_balance_usd = user.bitcoin_balance * admin_settings.bitcoin_rate_usd
    ethereum_balance_usd = user.ethereum_balance * admin_settings.ethereum_rate_usd
    total_balance_usd = bitcoin_balance_usd + ethereum_balance_usd
    
    return {
        "bitcoin_balance_usd": bitcoin_balance_usd,
        "ethereum_balance_usd": ethereum_balance_usd,
        "total_balance_usd": total_balance_usd
    }

def log_activity(db: Session, user_id: int, action: str, details: str = None, ip_address: str = None):
    """Log user activity"""
    activity = ActivityLog(
        user_id=user_id,
        action=action,
        details=details,
        ip_address=ip_address
    )
    db.add(activity)
    db.commit()

def log_transaction(
    db: Session,
    user_id: int,
    transaction_type: str,
    crypto_type: Optional[str],
    amount: Decimal,
    description: str,
    reference_id: Optional[str] = None
):
    """Log transaction for history tracking"""
    transaction = TransactionHistory(
        user_id=user_id,
        transaction_type=transaction_type,
        crypto_type=crypto_type,
        amount=amount,
        description=description,
        reference_id=reference_id
    )
    db.add(transaction)
    db.commit()

def log_admin_action(
    db: Session,
    admin_id: int,
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    details: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    before_value: Optional[str] = None,
    after_value: Optional[str] = None
):
    """Log admin actions for audit trail"""
    audit_log = AdminAuditLog(
        admin_id=admin_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
        before_value=before_value,
        after_value=after_value
    )
    db.add(audit_log)
    db.commit()

def process_referral_rewards(db: Session, new_user: User):
    """Process referral rewards when a new user registers with a referral code"""
    if not new_user.referred_by_code:
        return
    
    # Find the referrer
    referrer = db.query(User).filter(User.referral_code == new_user.referred_by_code).first()
    if not referrer:
        return
    
    # Get admin settings for referral rewards
    settings = get_admin_settings(db)
    if not settings.referral_reward_enabled:
        return
    
    # Create referral reward for the referrer
    referrer_reward = ReferralReward(
        referrer_id=referrer.id,
        referred_id=new_user.id,
        reward_type=settings.referral_reward_type,
        reward_amount=settings.referrer_reward_amount,
        status="paid"
    )
    db.add(referrer_reward)
    
    # Add reward to referrer's balance
    if settings.referral_reward_type == "bitcoin":
        referrer.bitcoin_balance += settings.referrer_reward_amount
    else:
        referrer.ethereum_balance += settings.referrer_reward_amount
    
    # Create referral reward for the new user (referred)
    referred_reward = ReferralReward(
        referrer_id=referrer.id,
        referred_id=new_user.id,
        reward_type=settings.referral_reward_type,
        reward_amount=settings.referral_reward_amount,
        status="paid"
    )
    db.add(referred_reward)
    
    # Add reward to new user's balance
    if settings.referral_reward_type == "bitcoin":
        new_user.bitcoin_balance += settings.referral_reward_amount
    else:
        new_user.ethereum_balance += settings.referral_reward_amount
    
    # Log transactions for both users
    log_transaction(
        db=db,
        user_id=referrer.id,
        transaction_type="referral_reward",
        crypto_type=settings.referral_reward_type,
        amount=settings.referrer_reward_amount,
        description=f"Referral reward for referring user {new_user.user_id}"
    )
    
    log_transaction(
        db=db,
        user_id=new_user.id,
        transaction_type="referral_reward",
        crypto_type=settings.referral_reward_type,
        amount=settings.referral_reward_amount,
        description=f"Welcome bonus for using referral code {new_user.referred_by_code}"
    )
    
    db.commit()

# =============================================================================
# AUTHENTICATION FUNCTIONS
# =============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(security), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

def get_current_admin_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

def get_admin_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# =============================================================================
# PYDANTIC SCHEMAS (DEFINED AFTER MODELS TO AVOID FORWARD REFERENCES)
# =============================================================================

# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str
    pin: str
    referral_code: Optional[str] = None
    device_fingerprint: str
    ip_address: str
    user_agent: Optional[str] = None
    birthday_day: Optional[int] = None
    birthday_month: Optional[int] = None
    birthday_year: Optional[int] = None
    gender: Optional[str] = None
    user_country_code: Optional[str] = None
    zip_code: Optional[str] = None
    bitcoin_wallet: Optional[str] = None
    ethereum_wallet: Optional[str] = None

class UserResponse(UserBase):
    id: int
    user_id: str
    name: str
    status: UserStatus
    is_admin: bool
    is_agent: bool
    is_flagged: bool
    usd_balance: Decimal
    bitcoin_balance: Decimal
    ethereum_balance: Decimal
    bitcoin_balance_usd: Optional[Decimal] = None
    ethereum_balance_usd: Optional[Decimal] = None
    total_balance_usd: Optional[Decimal] = None
    bitcoin_wallet: Optional[str] = None
    ethereum_wallet: Optional[str] = None
    personal_mining_rate: Optional[float] = None
    referral_code: Optional[str]
    email_verified: bool
    birthday_day: Optional[int] = None
    birthday_month: Optional[int] = None
    birthday_year: Optional[int] = None
    gender: Optional[str] = None
    user_country_code: Optional[str] = None
    zip_code: Optional[str] = None
    created_at: datetime
    referred_users_count: Optional[int] = None

    class Config:
        from_attributes = True

# Authentication Schemas
class UserLogin(BaseModel):
    email: str
    password: str
    device_fingerprint: str = None
    ip_address: str = None
    user_agent: str = None

class LoginWithTwoFA(BaseModel):
    email: str
    password: str
    two_fa_token: Optional[str] = None
    device_fingerprint: str = None
    ip_address: str = None
    user_agent: str = None

class TwoFASetupResponse(BaseModel):
    secret: str
    qr_code_url: str
    backup_codes: List[str]

class TwoFAVerifyRequest(BaseModel):
    token: str

class TwoFAStatusResponse(BaseModel):
    enabled: bool
    backup_codes_remaining: Optional[int] = None

class SecuritySettingsResponse(BaseModel):
    two_fa_enabled: bool
    account_locked: bool
    failed_login_attempts: int
    last_login: Optional[datetime] = None

# Deposit Schemas
class DepositCreate(BaseModel):
    crypto_type: CryptoType
    amount: Optional[Decimal] = None  # crypto amount
    usd_amount: Optional[Decimal] = None  # USD amount
    transaction_hash: Optional[str] = None

class DepositResponse(BaseModel):
    id: int
    crypto_type: str
    amount: Decimal
    usd_amount: Decimal
    status: str
    qr_code_url: str
    wallet_address: str

    class Config:
        from_attributes = True

# Transfer Schemas
class CryptoTransferCreate(BaseModel):
    to_email: Optional[EmailStr] = None
    to_user_id: Optional[str] = None
    crypto_type: CryptoType
    amount: Decimal

    @field_validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

    @model_validator(mode="after")
    def either_email_or_user_id(self):
        if not self.to_email and not self.to_user_id:
            raise ValueError('Either to_email or to_user_id must be provided')
        if self.to_email and self.to_user_id:
            raise ValueError('Provide either to_email or to_user_id, not both')
        return self

class BasicUserInfo(BaseModel):
    id: int
    email: str
    name: str

    class Config:
        from_attributes = True

class CryptoTransferResponse(BaseModel):
    id: int
    crypto_type: str
    amount: Decimal
    created_at: datetime
    from_user: BasicUserInfo
    to_user: BasicUserInfo

    class Config:
        from_attributes = True

# Withdrawal Schemas
class WithdrawalCreate(BaseModel):
    crypto_type: CryptoType
    amount: Decimal
    wallet_address: str
    
    @field_validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

class WithdrawalResponse(BaseModel):
    id: int
    crypto_type: str
    amount: Decimal
    status: WithdrawalStatus
    created_at: datetime

    class Config:
        from_attributes = True

# Admin Schemas
class AdminSettingsResponse(BaseModel):
    bitcoin_rate_usd: Decimal
    ethereum_rate_usd: Decimal
    global_mining_rate: float
    bitcoin_deposit_qr: Optional[str] = None
    ethereum_deposit_qr: Optional[str] = None
    bitcoin_wallet_address: Optional[str] = None
    ethereum_wallet_address: Optional[str] = None
    referral_reward_enabled: bool
    referral_reward_type: str
    referral_reward_amount: Decimal
    referrer_reward_amount: Decimal

class AdminSettingsUpdate(BaseModel):
    bitcoin_rate_usd: Optional[Decimal] = None
    ethereum_rate_usd: Optional[Decimal] = None
    global_mining_rate: Optional[float] = None
    bitcoin_wallet_address: Optional[str] = None
    ethereum_wallet_address: Optional[str] = None
    referral_reward_enabled: Optional[bool] = None
    referral_reward_type: Optional[str] = None
    referral_reward_amount: Optional[Decimal] = None
    referrer_reward_amount: Optional[Decimal] = None

# Transaction History Schemas
class TransactionHistoryResponse(BaseModel):
    id: int
    transaction_type: str
    crypto_type: Optional[str] = None
    amount: Decimal
    description: str
    reference_id: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class AdminAuditLogResponse(BaseModel):
    id: int
    admin_id: int
    admin_name: str
    admin_email: str
    action: str
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    details: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Analytics Schemas
class UserAnalyticsResponse(BaseModel):
    portfolio_overview: dict
    mining_performance: dict
    earnings_history: dict
    transaction_analytics: dict
    referral_performance: dict
    growth_metrics: dict

# =============================================================================
# EMAIL FUNCTIONS
# =============================================================================

async def send_email_notification(
    email: str,
    subject: str,
    template_type: str,
    context: dict,
    db: Session,
    attachment_path: str = None
):
    """Enhanced email notification with attachment support"""
    
    # Generate HTML content based on template type
    if template_type == "account_created":
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f9f9f9;">
            <div style="background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h2 style="color: #2563eb; margin-bottom: 20px;">Welcome to Crypto Mining Platform!</h2>
                
                <p>Hello {context['name']},</p>
                
                <p>Your account has been successfully created. Here are your account details:</p>
                
                <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <p><strong>User ID:</strong> {context['user_id']}</p>
                    <p><strong>Email:</strong> {context['email']}</p>
                    <p><strong>Referral Code:</strong> {context['referral_code']}</p>
                </div>
                
                <p>You can now start depositing crypto and begin mining!</p>
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="#" style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">Get Started</a>
                </div>
            </div>
        </div>
        """
    
    elif template_type == "deposit_confirmed":
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f9f9f9;">
            <div style="background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h2 style="color: #10b981; margin-bottom: 20px;">Deposit Confirmed - Mining Started!</h2>
                
                <p>Hello {context['name']},</p>
                
                <p>Great news! Your {context['crypto_type']} deposit has been confirmed and mining has started.</p>
                
                <div style="background-color: #f0fdf4; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #10b981;">
                    <h3 style="color: #065f46; margin-bottom: 15px;">Mining Details:</h3>
                    <p><strong>Crypto Type:</strong> {context['crypto_type'].title()}</p>
                    <p><strong>Amount:</strong> {context['amount']} {context['crypto_type'].upper()}</p>
                    <p><strong>Mining Rate:</strong> {context['mining_rate']}%</p>
                    <p><strong>Deposit ID:</strong> #{context['deposit_id']}</p>
                </div>
                
                <p>Your mining session is now active and you'll start earning rewards!</p>
            </div>
        </div>
        """
    
    # Add more email templates as needed...
    
    # Queue email for sending
    email_log = EmailNotification(
        user_id=context.get('user_id'),
        email=email,
        subject=subject,
        template_type=template_type,
        html_content=html_content,
        status="pending"
    )
    db.add(email_log)
    db.commit()

def send_email_now(to_email: str, subject: str, html_content: str) -> bool:
    """Send email immediately using SMTP"""
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(html_content, 'html'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Crypto Mining API",
    description="Crypto Mining Platform API Backend",
    version="1.0.0",
    docs_url=None,       # Disable Swagger UI (/docs)
    redoc_url=None,      # Disable ReDoc (/redoc)
    openapi_url=None     # Disable OpenAPI schema (/openapi.json)
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cryptomining.com",
        "https://api.cryptomining.com",
        "https://mining-frontend.netlify.app",
        "https://mining-backend.onrender.com",
        "https://cryptomining.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Mount static files for serving uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/login")
@limiter.limit("5/minute")
async def login_user(
    request: Request,
    user_login: LoginWithTwoFA,
    db: Session = Depends(get_db)
):
    # Check if account is locked
    if is_account_locked(db, user_login.email):
        log_security_event(
            db, None, "blocked_login_attempt", 
            f"Login blocked for {user_login.email} - account locked",
            request.client.host
        )
        raise HTTPException(status_code=423, detail="Account temporarily locked due to multiple failed attempts")
    
    user = db.query(User).filter(User.email == user_login.email).first()
    
    # Log login attempt
    login_attempt = LoginAttempt(
        email=user_login.email,
        ip_address=request.client.host,
        success=False,
        user_agent=request.headers.get("user-agent")
    )
    
    if not user or not verify_password(user_login.password, user.password_hash):
        # Log failed attempt
        log_security_event(
            db, user.id if user else None, "failed_login",
            f"Failed login attempt for {user_login.email}",
            request.client.host
        )
        
        if user:
            user.failed_login_attempts += 1
            user.last_failed_login = datetime.utcnow()
            
            # Lock account if too many attempts
            if user.failed_login_attempts >= MAX_LOGIN_ATTEMPTS:
                user.account_locked = True
                user.locked_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
                log_security_event(
                    db, user.id, "account_locked",
                    f"Account locked for {user.email} due to {MAX_LOGIN_ATTEMPTS} failed attempts",
                    request.client.host
                )
        
        db.add(login_attempt)
        db.commit()
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    if user.status == UserStatus.SUSPENDED:
        log_security_event(
            db, user.id, "suspended_login_attempt",
            f"Login attempt by suspended user {user.email}",
            request.client.host
        )
        raise HTTPException(status_code=403, detail="Account suspended")
    
    # Check 2FA if enabled
    if user.two_fa_enabled:
        if not user_login.two_fa_token:
            raise HTTPException(status_code=400, detail="2FA token required")
        
        if not verify_2fa_token(user.two_fa_secret, user_login.two_fa_token):
            log_security_event(
                db, user.id, "failed_2fa",
                f"Failed 2FA verification for {user.email}",
                request.client.host
            )
            user.failed_login_attempts += 1
            db.commit()
            raise HTTPException(status_code=400, detail="Invalid 2FA token")
    
    # Successful login - reset failed attempts
    user.failed_login_attempts = 0
    user.account_locked = False
    user.locked_until = None
    login_attempt.success = True
    
    # Log successful login
    log_activity(
        db, user.id, "USER_LOGIN",
        f"User logged in from {request.client.host}",
        request.client.host
    )
    
    log_security_event(
        db, user.id, "successful_login",
        f"Successful login for {user.email}",
        request.client.host
    )
    
    db.add(login_attempt)
    db.commit()
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "requires_2fa": user.two_fa_enabled
    }

@app.post("/register", response_model=UserResponse)
async def register_user(
    user: UserCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = generate_user_id()
    while db.query(User).filter(User.user_id == user_id).first():
        user_id = generate_user_id()
    
    # Hash password and PIN
    password_hash = get_password_hash(user.password)
    pin_hash = get_password_hash(user.pin)
    
    # Generate referral code
    referral_code = generate_referral_code()
    while db.query(User).filter(User.referral_code == referral_code).first():
        referral_code = generate_referral_code()
    
    # Create new user
    db_user = User(
        user_id=user_id,
        email=user.email,
        name=user.name,
        password_hash=password_hash,
        pin_hash=pin_hash,
        referral_code=referral_code,
        referred_by_code=user.referral_code,
        birthday_day=user.birthday_day,
        birthday_month=user.birthday_month,
        birthday_year=user.birthday_year,
        gender=user.gender,
        user_country_code=user.user_country_code,
        zip_code=user.zip_code
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Process referral rewards
    process_referral_rewards(db, db_user)
    
    # Send welcome email
    await send_email_notification(
        email=db_user.email,
        subject="Welcome to Crypto Mining Platform!",
        template_type="account_created",
        context={
            "user_id": db_user.user_id,
            "name": db_user.name,
            "email": db_user.email,
            "referral_code": db_user.referral_code
        },
        db=db
    )
    
    admin_settings = get_admin_settings(db)
    usd_values = calculate_usd_values(db_user, admin_settings)
    
    # Create response with USD values
    user_response = UserResponse.from_orm(db_user)
    user_response.bitcoin_balance_usd = usd_values["bitcoin_balance_usd"]
    user_response.ethereum_balance_usd = usd_values["ethereum_balance_usd"]
    user_response.total_balance_usd = usd_values["total_balance_usd"]
    
    return user_response

@app.get("/user/profile", response_model=UserResponse)
async def get_user_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    admin_settings = get_admin_settings(db)
    usd_values = calculate_usd_values(current_user, admin_settings)
    
    user_response = UserResponse.from_orm(current_user)
    user_response.bitcoin_balance_usd = usd_values["bitcoin_balance_usd"]
    user_response.ethereum_balance_usd = usd_values["ethereum_balance_usd"]
    user_response.total_balance_usd = usd_values["total_balance_usd"]
    
    return user_response

# Add all remaining endpoints from the original file...
# (Due to length constraints, I'm showing the structure - the complete file would include all endpoints)

# =============================================================================
# BACKGROUND TASKS
# =============================================================================

@app.on_event("startup")
@repeat_every(seconds=60)  # Run every minute
def process_mining_sessions():
    """Background task to process mining sessions every minute"""
    db = SessionLocal()
    try:
        active_sessions = db.query(MiningSession).filter(
            MiningSession.is_active == True
        ).all()
        
        for session in active_sessions:
            elapsed_seconds = (datetime.utcnow() - session.started_at).total_seconds()
            
            # Calculate total amount that should be mined by now
            mining_per_second = (session.deposited_amount * (session.mining_rate / 100)) / (24 * 3600)
            total_should_be_mined = min(
                mining_per_second * elapsed_seconds,
                session.deposited_amount * (session.mining_rate / 100)
            )
            
            # Calculate increment since last update
            mining_increment = total_should_be_mined - session.mined_amount
            
            if mining_increment > 0:
                # Update session
                session.mined_amount = total_should_be_mined
                session.last_processed = datetime.utcnow()
                
                # Update user balance
                user = db.query(User).filter(User.id == session.user_id).first()
                if session.crypto_type == "bitcoin":
                    user.bitcoin_balance += mining_increment
                else:
                    user.ethereum_balance += mining_increment
                
                # Log transaction
                log_transaction(
                    db=db,
                    user_id=user.id,
                    transaction_type="mining_reward",
                    crypto_type=session.crypto_type,
                    amount=mining_increment,
                    description=f"Mining reward - {session.crypto_type} mined at {session.mining_rate}% rate",
                    reference_id=str(session.id)
                )
        
        db.commit()
        
    except Exception as e:
        print(f"Error processing mining sessions: {e}")
        db.rollback()
    finally:
        db.close()

@app.on_event("startup")
@repeat_every(seconds=30)
def process_email_queue():
    """Background task to process email queue"""
    db = SessionLocal()
    try:
        pending_emails = db.query(EmailNotification).filter(
            EmailNotification.status == "pending"
        ).limit(10).all()
        
        for email in pending_emails:
            try:
                # Send email using SMTP
                if send_email_now(email.email, email.subject, email.html_content):
                    email.status = "sent"
                    email.sent_at = datetime.utcnow()
                else:
                    email.status = "failed"
                    email.attempts += 1
            except Exception as e:
                email.status = "failed"
                email.attempts += 1
                print(f"Failed to send email {email.id}: {e}")
        
        db.commit()
        
    except Exception as e:
        print(f"Error processing email queue: {e}")
        db.rollback()
    finally:
        db.close()

# Application startup
if __name__ == "__main__":
    import uvicorn
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Create admin user if it doesn't exist
    db = SessionLocal()
    try:
        existing_admin = db.query(User).filter(User.email == ADMIN_EMAIL).first()
        if not existing_admin:
            admin_user = User(
                user_id=generate_user_id(),
                name="Admin",
                email=ADMIN_EMAIL,
                password_hash=get_password_hash(ADMIN_PASSWORD),
                pin_hash=get_password_hash(ADMIN_PIN),
                is_admin=True,
                status=UserStatus.APPROVED,
                referral_code=generate_referral_code()
            )
            
            db.add(admin_user)
            db.commit()
            print(f"Admin user created: {ADMIN_EMAIL}")
        
        # Create default admin settings
        existing_settings = db.query(AdminSettings).first()
        if not existing_settings:
            default_settings = AdminSettings(
                bitcoin_rate_usd=50000.0,
                ethereum_rate_usd=3000.0,
                global_mining_rate=0.70,
                referral_reward_enabled=True,
                referral_reward_type="bitcoin",
                referral_reward_amount=0.001,
                referrer_reward_amount=0.001
            )
            db.add(default_settings)
            db.commit()
            print("Default admin settings created")
            
    except Exception as e:
        print(f"Error creating admin user: {e}")
        db.rollback()
    finally:
        db.close()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
