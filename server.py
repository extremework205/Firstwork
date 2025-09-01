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

# Database imports
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text, UniqueConstraint, create_engine, and_, or_, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.types import DECIMAL as SQLDecimal
from sqlalchemy.orm import joinedload
from sqlalchemy import func
from fastapi import APIRouter  # Needed to create a router
from sqlalchemy import exists

# Authentication imports
from passlib.context import CryptContext
from jose import JWTError, jwt

# Email imports
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

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
# ENUMS
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
# DATABASE MODELS (Ordered to resolve forward references)
# =============================================================================

class AdminSettings(Base):
    __tablename__ = "admin_settings"
    
    id = Column(Integer, primary_key=True, index=True)
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

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(10), unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
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
    mining_sessions = relationship("MiningSession", back_populates="user")

class ReferralReward(Base):
    __tablename__ = "referral_rewards"
    
    id = Column(Integer, primary_key=True, index=True)
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
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    email = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    template_type = Column(String, nullable=False)  # deposit_confirmed, login_alert, etc.
    status = Column(String, default="pending")  # pending, sent, failed
    sent_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User")

class TransactionHistory(Base):
    __tablename__ = "transaction_history"
    
    id = Column(Integer, primary_key=True, index=True)
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
    
    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)
    target_type = Column(String, nullable=True)  # user, deposit, settings, etc.
    target_id = Column(String, nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    admin = relationship("User")

class CryptoDeposit(Base):
    __tablename__ = "crypto_deposits"
    
    id = Column(Integer, primary_key=True, index=True)
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

    # Relationships
    user = relationship("User", back_populates="deposits", foreign_keys=[user_id])
    confirmed_by_user = relationship("User", foreign_keys=[confirmed_by])

class MiningRate(Base):
    __tablename__ = "mining_rates"
    
    id = Column(Integer, primary_key=True, index=True)
    crypto_type = Column(String, nullable=False)  # bitcoin or ethereum
    global_rate = Column(Float, nullable=False)  # Global mining rate percentage
    duration_hours = Column(Integer, default=24)  # Duration for the rate
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class DepositQRCode(Base):
    __tablename__ = "deposit_qr_codes"
    
    id = Column(Integer, primary_key=True, index=True)
    crypto_type = Column(String, nullable=False)  # bitcoin or ethereum
    qr_code_url = Column(String, nullable=False)  # URL to uploaded QR code image
    wallet_address = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UserDevice(Base):
    __tablename__ = "user_devices"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    device_fingerprint = Column(String, nullable=False)
    ip_address = Column(String, nullable=False)
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="devices")
    
    __table_args__ = (UniqueConstraint('user_id', 'device_fingerprint', name='unique_user_device'),)

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)  # e.g., "USER_LOGIN", "DEPOSIT_MADE"
    details = Column(Text)
    ip_address = Column(String)
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="activity_logs")

class SecurityLog(Base):
    __tablename__ = "security_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    event_type = Column(String, nullable=False)  # failed_login, successful_login, 2fa_enabled, etc.
    details = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User")

class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False)
    ip_address = Column(String, nullable=False)
    success = Column(Boolean, nullable=False)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class FraudFlag(Base):
    __tablename__ = "fraud_flags"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reason = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="fraud_flag")

class OTP(Base):
    __tablename__ = "otps"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False)
    otp_code = Column(String, nullable=False)
    purpose = Column(String, nullable=False) # signup, password_reset, pin_reset
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CryptoTransfer(Base):
    __tablename__ = "crypto_transfers"
    
    id = Column(Integer, primary_key=True, index=True)
    from_user_id = Column(Integer, ForeignKey("users.id"))
    to_user_id = Column(Integer, ForeignKey("users.id"))
    crypto_type = Column(String, nullable=False)  # bitcoin or ethereum
    amount = Column(SQLDecimal(18, 8), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    from_user = relationship("User", foreign_keys=[from_user_id], back_populates="sent_transfers")
    to_user = relationship("User", foreign_keys=[to_user_id], back_populates="received_transfers")

class MiningSession(Base):
    __tablename__ = "mining_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
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

    user = relationship("User", back_populates="mining_sessions")

class Withdrawal(Base):
    __tablename__ = "withdrawals"
    
    id = Column(Integer, primary_key=True, index=True)
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
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    approved_by = Column(Integer, ForeignKey("users.id"))
    status = Column(String, nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", foreign_keys=[user_id], back_populates="approvals")

# =============================================================================
# UTILITY FUNCTIONS (Defined before use)
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

def log_transaction(db: Session, user_id: int, transaction_type: str, crypto_type: str, amount: Decimal, description: str, reference_id: str = None):
    """Log transaction history"""
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
        new_user.bitcoin_balance += settings.referrer_reward_amount
    else:
        new_user.ethereum_balance += settings.referrer_reward_amount
    
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
# AUTHENTICATION SETUP
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
# PYDANTIC SCHEMAS (Ordered to resolve forward references)
# =============================================================================

# Base schemas first
class BasicUserInfo(BaseModel):
    id: int
    email: str
    name: str

    class Config:
        from_attributes = True

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

class UserPinVerify(BaseModel):
    pin: str

# Password and PIN Schemas
class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

class PinChangeRequest(BaseModel):
    current_pin: str
    new_pin: str

class PasswordResetRequest(BaseModel):
    email: str
    otp_code: str
    new_password: str

class PinResetRequest(BaseModel):
    email: str
    otp_code: str
    new_pin: str

# 2FA Schemas
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

# OTP Schemas
class OTPRequest(BaseModel):
    email: EmailStr
    purpose: str

class OTPVerify(BaseModel):
    email: EmailStr
    otp_code: str
    purpose: str

# Crypto Transfer Schemas
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
        # self is the model instance
        if not self.to_email and not self.to_user_id:
            raise ValueError('Either to_email or to_user_id must be provided')
        if self.to_email and self.to_user_id:
            raise ValueError('Provide either to_email or to_user_id, not both')
        return self

class CryptoTransferResponse(BaseModel):
    id: int
    crypto_type: str
    amount: Decimal
    created_at: datetime
    from_user: BasicUserInfo
    to_user: BasicUserInfo

    class Config:
        from_attributes = True

# Deposit Schemas
class DepositCreate(BaseModel):
    crypto_type: CryptoType
    amount: Optional[Decimal] = None  # crypto amount
    usd_amount: Optional[Decimal] = None  # USD amount
    transaction_hash: Optional[str] = None

class DepositEvidenceUpload(BaseModel):
    deposit_id: int
    evidence_file: str  # base64 encoded image or file URL

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

class AdminWithdrawalResponse(BaseModel):
    id: int
    crypto_type: str
    amount: Decimal
    status: WithdrawalStatus
    created_at: datetime
    user_id: int
    user_email: str
    wallet_address: str

    class Config:
        from_attributes = True

# Mining Schemas
class MiningRateCreate(BaseModel):
    crypto_type: CryptoType
    global_rate: float
    duration_hours: int = 24

class MiningRateResponse(BaseModel):
    id: int
    crypto_type: str
    global_rate: float
    duration_hours: int
    is_active: bool

    class Config:
        from_attributes = True

# QR Code Schemas
class QRCodeCreate(BaseModel):
    crypto_type: CryptoType
    qr_code_url: str
    wallet_address: str

class QRCodeResponse(BaseModel):
    id: int
    crypto_type: str
    qr_code_url: str
    wallet_address: str
    is_active: bool

    class Config:
        from_attributes = True

# Admin Schemas
class UserStatusUpdate(BaseModel):
    status: UserStatus

class AgentAssignment(BaseModel):
    user_id: int
    is_agent: bool

class SystemSettingUpdate(BaseModel):
    key: str
    value: str
    description: Optional[str] = None

class BulkUserStatusUpdate(BaseModel):
    user_ids: List[int]
    status: UserStatus

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

class AdminCryptoTransferRequest(BaseModel):
    from_user_email: Optional[EmailStr] = None
    to_user_email: EmailStr
    crypto_type: CryptoType
    amount: Decimal

    @field_validator("amount")
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Amount must be greater than 0")
        return v

# Transaction and History Schemas
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

class TransactionFilter(BaseModel):
    transaction_type: Optional[str] = None
    crypto_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_amount: Optional[Decimal] = None
    max_amount: Optional[Decimal] = None

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

# Referral Schemas
class ReferralRewardResponse(BaseModel):
    id: int
    referrer_id: int
    referred_id: int
    reward_type: str
    reward_amount: Decimal
    status: str
    created_at: datetime
    paid_at: Optional[datetime] = None

# Fraud and Security Schemas
class FraudFlagResponse(BaseModel):
    user: BasicUserInfo
    reason: str
    created_at: datetime

    class Config:
        orm_mode = True

class FraudRuleUpdate(BaseModel):
    rule_key: str
    limit_value: int
    action: str
    description: Optional[str] = None

class FraudRulePatch(BaseModel):
    limit_value: int
    action: str

# Dashboard and Analytics Schemas
class DashboardStats(BaseModel):
    usd_balance: Decimal
    bitcoin_balance: Decimal
    ethereum_balance: Decimal
    active_mining_sessions: int
    total_deposits: int
    pending_withdrawals: int
    is_flagged: bool

class AdminDashboardStats(BaseModel):
    total_users: int
    total_deposits: int
    total_crypto_distributed: Decimal
    pending_withdrawals: int
    active_mining_sessions: int

class TransferHistory(BaseModel):
    transfers: List[CryptoTransferResponse]

class UserAnalyticsResponse(BaseModel):
    portfolio_overview: dict
    mining_performance: dict
    earnings_history: dict
    transaction_analytics: dict
    referral_performance: dict
    growth_metrics: dict

class MiningPerformanceData(BaseModel):
    daily_earnings: List[dict]
    weekly_summary: dict
    monthly_trends: dict
    efficiency_metrics: dict

class PortfolioAnalytics(BaseModel):
    current_value: dict
    historical_performance: List[dict]
    asset_allocation: dict
    growth_rate: dict

class EarningsBreakdown(BaseModel):
    mining_earnings: Decimal
    referral_earnings: Decimal
    total_earnings: Decimal
    earnings_by_period: List[dict]

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

env = Environment(loader=FileSystemLoader('template'))

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

# =============================================================================
# AUTHENTICATION
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
# ENHANCED AUTHENTICATION ENDPOINTS
# =============================================================================

@app.post("/login")
@limiter.limit("5/minute")  # Added rate limiting to login
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

@app.post("/security/2fa/setup", response_model=TwoFASetupResponse)
async def setup_2fa(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Setup 2FA for user account"""
    if current_user.two_fa_enabled:
        raise HTTPException(status_code=400, detail="2FA is already enabled")
    
    # Generate new secret
    secret = generate_2fa_secret()
    qr_url = get_2fa_qr_url(secret, current_user.email)
    
    # Generate backup codes
    backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
    
    # Store secret temporarily (not enabled until verified)
    current_user.two_fa_secret = secret
    db.commit()
    
    log_security_event(
        db, current_user.id, "2fa_setup_initiated",
        "User initiated 2FA setup",
        "system"
    )
    
    return TwoFASetupResponse(
        secret=secret,
        qr_code_url=qr_url,
        backup_codes=backup_codes
    )

@app.post("/security/2fa/verify")
async def verify_2fa_setup(
    verify_request: TwoFAVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verify and enable 2FA"""
    if not current_user.two_fa_secret:
        raise HTTPException(status_code=400, detail="2FA setup not initiated")
    
    if not verify_2fa_token(current_user.two_fa_secret, verify_request.token):
        raise HTTPException(status_code=400, detail="Invalid 2FA token")
    
    # Enable 2FA
    current_user.two_fa_enabled = True
    db.commit()
    
    log_security_event(
        db, current_user.id, "2fa_enabled",
        "User successfully enabled 2FA",
        "system"
    )
    
    return {"message": "2FA enabled successfully"}

@app.post("/security/2fa/disable")
async def disable_2fa(
    verify_request: TwoFAVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Disable 2FA"""
    if not current_user.two_fa_enabled:
        raise HTTPException(status_code=400, detail="2FA is not enabled")
    
    if not verify_2fa_token(current_user.two_fa_secret, verify_request.token):
        raise HTTPException(status_code=400, detail="Invalid 2FA token")
    
    # Disable 2FA
    current_user.two_fa_enabled = False
    current_user.two_fa_secret = None
    db.commit()
    
    log_security_event(
        db, current_user.id, "2fa_disabled",
        "User disabled 2FA",
        "system"
    )
    
    return {"message": "2FA disabled successfully"}

@app.get("/security/2fa/status", response_model=TwoFAStatusResponse)
async def get_2fa_status(
    current_user: User = Depends(get_current_user)
):
    """Get 2FA status"""
    return TwoFAStatusResponse(
        enabled=current_user.two_fa_enabled
    )

@app.get("/security/settings", response_model=SecuritySettingsResponse)
async def get_security_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user security settings"""
    # Get last successful login
    last_login = db.query(SecurityLog).filter(
        SecurityLog.user_id == current_user.id,
        SecurityLog.event_type == "successful_login"
    ).order_by(SecurityLog.created_at.desc()).first()
    
    return SecuritySettingsResponse(
        two_fa_enabled=current_user.two_fa_enabled,
        account_locked=current_user.account_locked,
        failed_login_attempts=current_user.failed_login_attempts,
        last_login=last_login.created_at if last_login else None
    )

@app.get("/admin/security/logs")
async def get_security_logs(
    limit: int = 100,
    offset: int = 0,
    event_type: Optional[str] = None,
    user_email: Optional[str] = None,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get security logs for admin"""
    query = db.query(SecurityLog).options(joinedload(SecurityLog.user))
    
    if event_type:
        query = query.filter(SecurityLog.event_type == event_type)
    
    if user_email:
        query = query.join(User).filter(User.email.ilike(f"%{user_email}%"))
    
    logs = query.order_by(SecurityLog.created_at.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()
    
    return [
        {
            "id": log.id,
            "user_email": log.user.email if log.user else "Unknown",
            "event_type": log.event_type,
            "details": log.details,
            "ip_address": log.ip_address,
            "created_at": log.created_at
        }
        for log in logs
    ]

@app.post("/admin/security/unlock-account")
async def unlock_user_account(
    user_email: str,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Unlock a user account"""
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.account_locked = False
    user.locked_until = None
    user.failed_login_attempts = 0
    
    log_security_event(
        db, user.id, "account_unlocked",
        f"Account unlocked by admin {admin_user.email}",
        "admin"
    )
    
    db.commit()
    return {"message": f"Account unlocked for {user_email}"}

@app.get("/admin/security/stats")
async def get_security_stats(
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get security statistics"""
    # Get stats for last 24 hours
    yesterday = datetime.utcnow() - timedelta(days=1)
    
    failed_logins = db.query(SecurityLog).filter(
        SecurityLog.event_type == "failed_login",
        SecurityLog.created_at >= yesterday
    ).count()
    
    successful_logins = db.query(SecurityLog).filter(
        SecurityLog.event_type == "successful_login",
        SecurityLog.created_at >= yesterday
    ).count()
    
    locked_accounts = db.query(User).filter(User.account_locked == True).count()
    
    users_with_2fa = db.query(User).filter(User.two_fa_enabled == True).count()
    total_users = db.query(User).count()
    
    return {
        "last_24_hours": {
            "failed_logins": failed_logins,
            "successful_logins": successful_logins
        },
        "current_status": {
            "locked_accounts": locked_accounts,
            "users_with_2fa": users_with_2fa,
            "total_users": total_users,
            "2fa_adoption_rate": (users_with_2fa / total_users * 100) if total_users > 0 else 0
        }
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
    password_hash = pwd_context.hash(user.password)
    pin_hash = pwd_context.hash(user.pin)
    
    # Generate referral code
    referral_code = generate_referral_code()
    while db.query(User).filter(User.referral_code == referral_code).first():
        referral_code = generate_referral_code()
    
    # Create new user
    db_user = User(
        user_id=user_id,  # Added user_id
        email=user.email,
        name=user.name,
        password_hash=password_hash,
        pin_hash=pin_hash,
        referral_code=referral_code,
        referred_by_code=user.referred_by_code,
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
    
    process_referral_rewards(db, db_user)
    
    await send_email_notification(
        email=db_user.email,
        subject="Welcome to Crypto Mining Platform!",
        template_type="account_created",
        context={
            "user_id": db_user.id,
            "name": db_user.name,
            "email": db_user.email,
            "user_id": db_user.user_id,
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

# =============================================================================
# CRYPTO DEPOSIT ENDPOINTS
# =============================================================================

def get_or_create_admin_settings(db: Session):
    """Get or create admin settings"""
    settings = db.query(AdminSettings).first()
    if not settings:
        settings = AdminSettings()
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings

@app.put("/api/admin/wallet-addresses")
def update_wallet_addresses(
    wallet_data: dict,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Update Bitcoin and Ethereum wallet addresses"""
    settings = get_or_create_admin_settings(db)
    
    if "bitcoin_wallet_address" in wallet_data:
        old_bitcoin = settings.bitcoin_wallet_address
        settings.bitcoin_wallet_address = wallet_data["bitcoin_wallet_address"]
        log_admin_action(
            db=db,
            admin_id=admin_user.id,
            action="update_bitcoin_wallet",
            target_type="admin_settings",
            target_id="1",
            details=f"Updated Bitcoin wallet address from {old_bitcoin} to {wallet_data['bitcoin_wallet_address']}"
        )
    
    if "ethereum_wallet_address" in wallet_data:
        old_ethereum = settings.ethereum_wallet_address
        settings.ethereum_wallet_address = wallet_data["ethereum_wallet_address"]
        log_admin_action(
            db=db,
            admin_id=admin_user.id,
            action="update_ethereum_wallet",
            target_type="admin_settings",
            target_id="1",
            details=f"Updated Ethereum wallet address from {old_ethereum} to {wallet_data['ethereum_wallet_address']}"
        )
    
    db.commit()
    db.refresh(settings)
    
    return {
        "message": "Wallet addresses updated successfully",
        "bitcoin_wallet_address": settings.bitcoin_wallet_address,
        "ethereum_wallet_address": settings.ethereum_wallet_address
    }

@app.get("/api/admin/wallet-addresses")
def get_wallet_addresses(admin_user: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Get current wallet addresses"""
    settings = get_or_create_admin_settings(db)
    return {
        "bitcoin_wallet_address": settings.bitcoin_wallet_address,
        "ethereum_wallet_address": settings.ethereum_wallet_address
    }

@app.get("/api/deposits/info/{crypto_type}")
def get_deposit_info(crypto_type: CryptoType, db: Session = Depends(get_db)):
    """Get deposit information including QR code and wallet address"""
    settings = get_or_create_admin_settings(db)
    
    # Get wallet address from admin settings
    if crypto_type == "bitcoin":
        wallet_address = settings.bitcoin_wallet_address
        qr_code_url = settings.bitcoin_deposit_qr
    else:  # ethereum
        wallet_address = settings.ethereum_wallet_address
        qr_code_url = settings.ethereum_deposit_qr
    
    if not wallet_address:
        raise HTTPException(status_code=404, detail=f"No wallet address configured for {crypto_type}")
    
    return {
        "crypto_type": crypto_type,
        "qr_code_url": qr_code_url,
        "wallet_address": wallet_address
    }

def get_crypto_usd_rate(db: Session, crypto_type: str) -> Decimal:
    """Get current USD rate for crypto from admin settings"""
    if crypto_type == "bitcoin":
        rate_setting = db.query(AdminSettings).filter(AdminSettings.key == "bitcoin_usd_rate").first()
    else:
        rate_setting = db.query(AdminSettings).filter(AdminSettings.key == "ethereum_usd_rate").first()
    
    if rate_setting:
        return Decimal(rate_setting.value)
    
    # Default rates if not set by admin
    return Decimal("50000.00") if crypto_type == "bitcoin" else Decimal("3000.00")

def convert_crypto_to_usd(crypto_amount: Decimal, crypto_type: str, db: Session) -> Decimal:
    """Convert crypto amount to USD"""
    rate = get_crypto_usd_rate(db, crypto_type)
    return crypto_amount * rate

def convert_usd_to_crypto(usd_amount: Decimal, crypto_type: str, db: Session) -> Decimal:
    """Convert USD amount to crypto"""
    rate = get_crypto_usd_rate(db, crypto_type)
    return usd_amount / rate

@app.get("/api/deposits/rates")
def get_crypto_rates(db: Session = Depends(get_db)):
    """Get current crypto to USD conversion rates"""
    bitcoin_rate = get_crypto_usd_rate(db, "bitcoin")
    ethereum_rate = get_crypto_usd_rate(db, "ethereum")
    
    return {
        "bitcoin_usd_rate": bitcoin_rate,
        "ethereum_usd_rate": ethereum_rate
    }

@app.post("/api/deposits/convert")
def convert_amount(
    crypto_type: CryptoType,
    amount: Optional[Decimal] = None,
    usd_amount: Optional[Decimal] = None,
    db: Session = Depends(get_db)
):
    """Convert between crypto and USD amounts"""
    if amount and usd_amount:
        raise HTTPException(status_code=400, detail="Provide either crypto amount or USD amount, not both")
    
    if not amount and not usd_amount:
        raise HTTPException(status_code=400, detail="Provide either crypto amount or USD amount")
    
    if amount:
        # Convert crypto to USD
        usd_equivalent = convert_crypto_to_usd(amount, crypto_type, db)
        return {
            "crypto_amount": amount,
            "usd_amount": usd_equivalent,
            "crypto_type": crypto_type
        }
    else:
        # Convert USD to crypto
        crypto_equivalent = convert_usd_to_crypto(usd_amount, crypto_type, db)
        return {
            "crypto_amount": crypto_equivalent,
            "usd_amount": usd_amount,
            "crypto_type": crypto_type
        }

@app.post("/api/deposits/create")
def create_deposit(
    deposit_data: DepositCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new crypto deposit with USD conversion"""
    if current_user.is_flagged:
        raise HTTPException(status_code=403, detail="Account flagged - deposits not allowed")
    
    # Validate input - must provide either crypto amount or USD amount
    if deposit_data.amount and deposit_data.usd_amount:
        raise HTTPException(status_code=400, detail="Provide either crypto amount or USD amount, not both")
    
    if not deposit_data.amount and not deposit_data.usd_amount:
        raise HTTPException(status_code=400, detail="Provide either crypto amount or USD amount")
    
    # Calculate both amounts
    if deposit_data.amount:
        crypto_amount = deposit_data.amount
        usd_amount = convert_crypto_to_usd(crypto_amount, deposit_data.crypto_type, db)
    else:
        usd_amount = deposit_data.usd_amount
        crypto_amount = convert_usd_to_crypto(usd_amount, deposit_data.crypto_type, db)
    
    # Get deposit info (QR code and wallet address)
    qr_info = db.query(DepositQRCode).filter(
        DepositQRCode.crypto_type == deposit_data.crypto_type,
        DepositQRCode.is_active == True
    ).first()
    
    if not qr_info:
        raise HTTPException(status_code=404, detail=f"No deposit info available for {deposit_data.crypto_type}")
    
    deposit = CryptoDeposit(
        user_id=current_user.id,
        crypto_type=deposit_data.crypto_type,
        amount=crypto_amount,
        usd_amount=usd_amount,
        transaction_hash=deposit_data.transaction_hash,
        status=DepositStatus.PENDING
    )
    
    db.add(deposit)
    db.commit()
    db.refresh(deposit)
    
    log_activity(
        db, current_user.id, "DEPOSIT_CREATED",
        f"Created {deposit_data.crypto_type} deposit of {crypto_amount} (${usd_amount} USD)",
        request.client.host
    )
    
    return {
        "message": "Deposit created successfully",
        "deposit_id": deposit.id,
        "crypto_amount": crypto_amount,
        "usd_amount": usd_amount,
        "qr_code_url": qr_info.qr_code_url,
        "wallet_address": qr_info.wallet_address,
        "crypto_type": deposit_data.crypto_type
    }

@app.post("/api/deposits/{deposit_id}/upload-evidence")
async def upload_deposit_evidence(
    deposit_id: int,
    evidence_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload deposit evidence (screenshot/proof)"""
    deposit = db.query(CryptoDeposit).filter(
        CryptoDeposit.id == deposit_id,
        CryptoDeposit.user_id == current_user.id
    ).first()
    
    if not deposit:
        raise HTTPException(status_code=404, detail="Deposit not found")
    
    if deposit.status != DepositStatus.PENDING:
        raise HTTPException(status_code=400, detail="Can only upload evidence for pending deposits")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif", "application/pdf"]
    if evidence_file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only images (JPEG, PNG, GIF) and PDF files are allowed")
    
    # Save file (in production, use cloud storage like AWS S3)
    import os
    import uuid
    
    upload_dir = "uploads/deposit_evidence"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_extension = evidence_file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    with open(file_path, "wb") as buffer:
        content = await evidence_file.read()
        buffer.write(content)
    
    # Update deposit with evidence URL
    deposit.evidence_url = f"/uploads/deposit_evidence/{unique_filename}"
    db.commit()
    
    # Get admin email for forwarding evidence
    admin_email_setting = db.query(AdminSettings).filter(AdminSettings.key == "admin_email").first()
    admin_email = admin_email_setting.value if admin_email_setting else "admin@example.com"
    
    # Send evidence to admin email
    await send_email_notification(
        email=admin_email,
        subject=f"New Deposit Evidence - {deposit.crypto_type.title()} Deposit #{deposit.id}",
        template_type="deposit_evidence",
        context={
            "user_email": current_user.email,
            "user_name": current_user.name,
            "crypto_type": deposit.crypto_type.title(),
            "amount": deposit.amount,
            "usd_amount": deposit.usd_amount,
            "deposit_id": deposit.id,
            "evidence_url": deposit.evidence_url,
            "transaction_hash": deposit.transaction_hash or "Not provided"
        },
        db=db,
        attachment_path=file_path  # Attach the evidence file
    )
    
    log_activity(
        db, current_user.id, "DEPOSIT_EVIDENCE_UPLOADED",
        f"Uploaded evidence for {deposit.crypto_type} deposit #{deposit.id}",
        "system"
    )
    
    return {
        "message": "Evidence uploaded successfully",
        "evidence_url": deposit.evidence_url
    }

@app.post("/api/deposits/{deposit_id}/submit")
async def submit_deposit(
    deposit_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit deposit for admin review (with or without evidence)"""
    deposit = db.query(CryptoDeposit).filter(
        CryptoDeposit.id == deposit_id,
        CryptoDeposit.user_id == current_user.id
    ).first()
    
    if not deposit:
        raise HTTPException(status_code=404, detail="Deposit not found")
    
    if deposit.status != DepositStatus.PENDING:
        raise HTTPException(status_code=400, detail="Deposit already submitted")
    
    # Get admin email for notification
    admin_email_setting = db.query(AdminSettings).filter(AdminSettings.key == "admin_email").first()
    admin_email = admin_email_setting.value if admin_email_setting else "admin@example.com"
    
    # Send notification to admin about new deposit submission
    await send_email_notification(
        email=admin_email,
        subject=f"New Deposit Submission - {deposit.crypto_type.title()} #{deposit.id}",
        template_type="deposit_submission",
        context={
            "user_email": current_user.email,
            "user_name": current_user.name,
            "crypto_type": deposit.crypto_type.title(),
            "amount": deposit.amount,
            "usd_amount": deposit.usd_amount,
            "deposit_id": deposit.id,
            "has_evidence": bool(deposit.evidence_url),
            "evidence_url": deposit.evidence_url,
            "transaction_hash": deposit.transaction_hash or "Not provided"
        },
        db=db
    )
    
    log_activity(
        db, current_user.id, "DEPOSIT_SUBMITTED",
        f"Submitted {deposit.crypto_type} deposit #{deposit.id} for admin review",
        "system"
    )
    
    return {"message": "Deposit submitted for admin review"}

# =============================================================================
# ADMIN DEPOSIT MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/admin/deposits/pending")
def get_pending_deposits(admin_user: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Get all pending deposits for admin confirmation"""
    deposits = db.query(CryptoDeposit).options(joinedload(CryptoDeposit.user)).filter(
        CryptoDeposit.status == DepositStatus.PENDING
    ).order_by(CryptoDeposit.created_at.desc()).all()
    
    return [
        {
            "id": d.id,
            "user_email": d.user.email,
            "crypto_type": d.crypto_type,
            "amount": d.amount,
            "transaction_hash": d.transaction_hash,
            "created_at": d.created_at
        }
        for d in deposits
    ]

@app.put("/api/admin/deposits/{deposit_id}/confirm")
async def confirm_deposit(
    deposit_id: int,
    request: Request,
    action: str = Query(...),  # "confirm" or "reject"
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Confirm or reject a deposit and start mining if confirmed"""
    deposit = db.query(CryptoDeposit).filter(CryptoDeposit.id == deposit_id).first()
    if not deposit:
        raise HTTPException(status_code=404, detail="Deposit not found")
    
    user = db.query(User).filter(User.id == deposit.user_id).first()
    
    if action == "confirm":
        deposit.status = DepositStatus.CONFIRMED
        deposit.confirmed_by = admin_user.id
        deposit.confirmed_at = datetime.utcnow()
        
        # Add crypto to user balance
        if deposit.crypto_type == "bitcoin":
            user.bitcoin_balance += deposit.amount
        else:
            user.ethereum_balance += deposit.amount
        
        # Get mining rate (personal or global)
        mining_rate = user.personal_mining_rate
        if not mining_rate:
            rate_setting = db.query(MiningRate).filter(
                MiningRate.crypto_type == deposit.crypto_type,
                MiningRate.is_active == True
            ).first()
            mining_rate = rate_setting.global_rate if rate_setting else 70.0  # Default 70%
        
        # Create mining session
        mining_session = MiningSession(
            user_id=user.id,
            deposit_id=deposit.id,
            crypto_type=deposit.crypto_type,
            deposited_amount=deposit.amount,
            mining_rate=mining_rate,
            is_active=True
        )
        
        log_transaction(
            db=db,
            user_id=user.id,
            transaction_type="deposit",
            crypto_type=deposit.crypto_type,
            amount=deposit.amount,
            description=f"Deposit confirmed by admin - {deposit.crypto_type} mining started at {mining_rate}%",
            reference_id=str(deposit.id)
        )
        
        log_admin_action(
            db=db,
            admin_id=admin_user.id,
            action="confirm_deposit",
            target_type="deposit",
            target_id=str(deposit.id),
            description=f"Confirmed {deposit.crypto_type} deposit of {deposit.amount} for user {user.email}",
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", "")
        )
        
        # Send confirmation email
        await send_email_notification(
            email=user.email,
            subject="Deposit Confirmed - Mining Started!",
            template_type="deposit_confirmed",
            context={
                "name": user.name,
                "crypto_type": deposit.crypto_type.title(),
                "amount": deposit.amount,
                "mining_rate": mining_rate,
                "deposit_id": deposit.id
            },
            db=db
        )
        
    elif action == "reject":
        deposit.status = DepositStatus.REJECTED
        deposit.confirmed_by = admin_user.id
        deposit.confirmed_at = datetime.utcnow()
        
        log_admin_action(
            db=db,
            admin_id=admin_user.id,
            action="reject_deposit",
            target_type="deposit",
            target_id=str(deposit.id),
            description=f"Rejected {deposit.crypto_type} deposit of {deposit.amount} for user {user.email}",
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", "")
        )
    
    db.commit()
    
    return {"message": f"Deposit {action}ed successfully"}

# =============================================================================
# MINING ENDPOINTS
# =============================================================================

@app.get("/api/mining/live-progress")
def get_live_mining_progress(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get real-time mining progress for all active sessions"""
    active_sessions = db.query(MiningSession).filter(
        MiningSession.user_id == current_user.id,
        MiningSession.is_active == True
    ).all()
    
    progress_data = []
    for session in active_sessions:
        elapsed_seconds = (datetime.utcnow() - session.created_at).total_seconds()
        
        # Calculate mining per second (rate over 24 hours)
        mining_per_second = (session.deposited_amount * (session.mining_rate / 100)) / (24 * 3600)
        
        # Calculate current mined amount
        current_mined = min(
            mining_per_second * elapsed_seconds,
            session.deposited_amount * (session.mining_rate / 100)
        )
        
        progress_data.append({
            "session_id": session.id,
            "crypto_type": session.crypto_type,
            "deposited_amount": session.deposited_amount,
            "mining_rate": session.mining_rate,
            "current_mined": current_mined,
            "mining_per_second": mining_per_second,
            "progress_percentage": (current_mined / (session.deposited_amount * (session.mining_rate / 100))) * 100,
            "elapsed_hours": elapsed_seconds / 3600
        })
    
    return {"active_sessions": progress_data}

@app.put("/api/admin/mining/{session_id}/pause")
def pause_mining_session(
    session_id: int,
    request: Request,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Pause a mining session"""
    session = db.query(MiningSession).filter(MiningSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Mining session not found")
    
    session.is_active = False
    session.paused_at = datetime.utcnow()
    
    log_admin_action(
        db=db,
        admin_id=admin_user.id,
        action="pause_mining",
        target_type="mining_session",
        target_id=str(session.id),
        description=f"Paused mining session for user {session.user.email}",
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent", "")
    )
    
    db.commit()
    
    return {"message": "Mining session paused successfully"}

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
            elapsed_seconds = (datetime.utcnow() - session.created_at).total_seconds()
            
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
@repeat_every(seconds=30)  # Process email queue every 30 seconds
def process_email_queue():
    """Background task to process email queue"""
    db = SessionLocal()
    try:
        pending_emails = db.query(EmailNotification).filter(
            EmailNotification.status == "pending"
        ).limit(10).all()
        
        for email in pending_emails:
            try:
                success = send_email_now(
                    to_email=email.recipient_email,
                    subject=email.subject,
                    html_content=email.html_content
                )
                
                if success:
                    email.status = "sent"
                    email.sent_at = datetime.utcnow()
                else:
                    email.status = "failed"
                    email.attempts += 1
                    
                    # Retry failed emails up to 3 times
                    if email.attempts < 3:
                        email.status = "pending"
                        email.scheduled_for = datetime.utcnow() + timedelta(minutes=5)
                
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

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Create admin user if it doesn't exist
    db = SessionLocal()
    try:
        admin_email = os.getenv("ADMIN_EMAIL")
        admin_password = os.getenv("ADMIN_PASSWORD")
        admin_pin = os.getenv("ADMIN_PIN")
        
        existing_admin = db.query(User).filter(User.email == admin_email).first()
        if not existing_admin:
            hashed_password = pwd_context.hash(admin_password)
            hashed_pin = pwd_context.hash(admin_pin)
            
            admin_user = User(
                name="Admin",
                email=admin_email,
                password=hashed_password,
                pin=hashed_pin,
                is_admin=True,
                is_verified=True,
                user_id=generate_user_id(),
                referral_code=generate_referral_code()
            )
            
            db.add(admin_user)
            db.commit()
            print(f"Admin user created: {admin_email}")
        
        # Create default admin settings
        existing_settings = db.query(AdminSettings).first()
        if not existing_settings:
            default_settings = AdminSettings(
                bitcoin_rate_usd=50000.0,
                ethereum_rate_usd=3000.0,
                referral_reward_bitcoin=0.001,
                referral_reward_ethereum=0.01,
                referee_reward_bitcoin=0.0005,
                referee_reward_ethereum=0.005
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

@app.get("/api/analytics/dashboard", response_model=UserAnalyticsResponse)
async def get_user_analytics_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive analytics dashboard for user"""
    admin_settings = get_admin_settings(db)
    
    # Portfolio Overview
    usd_values = calculate_usd_values(current_user, admin_settings)
    portfolio_overview = {
        "total_value_usd": usd_values["total_balance_usd"],
        "bitcoin_balance": current_user.bitcoin_balance,
        "ethereum_balance": current_user.ethereum_balance,
        "bitcoin_value_usd": usd_values["bitcoin_balance_usd"],
        "ethereum_value_usd": usd_values["ethereum_balance_usd"],
        "asset_allocation": {
            "bitcoin_percentage": (usd_values["bitcoin_balance_usd"] / usd_values["total_balance_usd"] * 100) if usd_values["total_balance_usd"] > 0 else 0,
            "ethereum_percentage": (usd_values["ethereum_balance_usd"] / usd_values["total_balance_usd"] * 100) if usd_values["total_balance_usd"] > 0 else 0
        }
    }
    
    # Mining Performance
    active_sessions = db.query(MiningSession).filter(
        MiningSession.user_id == current_user.id,
        MiningSession.is_active == True
    ).all()
    
    total_mining_power = sum(session.deposited_amount for session in active_sessions)
    avg_mining_rate = sum(session.mining_rate for session in active_sessions) / len(active_sessions) if active_sessions else 0
    
    mining_performance = {
        "active_sessions": len(active_sessions),
        "total_mining_power": total_mining_power,
        "average_mining_rate": avg_mining_rate,
        "daily_estimated_earnings": sum(
            (session.deposited_amount * (session.mining_rate / 100)) / 24 
            for session in active_sessions
        )
    }
    
    # Earnings History (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    earnings_transactions = db.query(TransactionHistory).filter(
        TransactionHistory.user_id == current_user.id,
        TransactionHistory.transaction_type.in_(["mining_reward", "referral_reward"]),
        TransactionHistory.created_at >= thirty_days_ago
    ).all()
    
    mining_earnings = sum(t.amount for t in earnings_transactions if t.transaction_type == "mining_reward")
    referral_earnings = sum(t.amount for t in earnings_transactions if t.transaction_type == "referral_reward")
    
    earnings_history = {
        "total_earnings": mining_earnings + referral_earnings,
        "mining_earnings": mining_earnings,
        "referral_earnings": referral_earnings,
        "daily_breakdown": []
    }
    
    # Daily earnings breakdown
    for i in range(30):
        day = datetime.utcnow() - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_earnings = db.query(func.sum(TransactionHistory.amount)).filter(
            TransactionHistory.user_id == current_user.id,
            TransactionHistory.transaction_type.in_(["mining_reward", "referral_reward"]),
            TransactionHistory.created_at >= day_start,
            TransactionHistory.created_at < day_end
        ).scalar() or 0
        
        earnings_history["daily_breakdown"].append({
            "date": day_start.strftime("%Y-%m-%d"),
            "earnings": day_earnings
        })
    
    # Transaction Analytics
    all_transactions = db.query(TransactionHistory).filter(
        TransactionHistory.user_id == current_user.id
    ).all()
    
    transaction_analytics = {
        "total_transactions": len(all_transactions),
        "transaction_types": {},
        "volume_by_crypto": {"bitcoin": 0, "ethereum": 0}
    }
    
    for transaction in all_transactions:
        # Count by type
        if transaction.transaction_type in transaction_analytics["transaction_types"]:
            transaction_analytics["transaction_types"][transaction.transaction_type] += 1
        else:
            transaction_analytics["transaction_types"][transaction.transaction_type] = 1
        
        # Volume by crypto
        if transaction.crypto_type:
            transaction_analytics["volume_by_crypto"][transaction.crypto_type] += transaction.amount
    
    # Referral Performance
    referrals = db.query(User).filter(User.referred_by_code == current_user.referral_code).all()
    referral_rewards = db.query(ReferralReward).filter(
        ReferralReward.referrer_id == current_user.id
    ).all()
    
    referral_performance = {
        "total_referrals": len(referrals),
        "total_rewards": sum(r.reward_amount for r in referral_rewards),
        "active_referrals": len([r for r in referrals if r.status == "approved"]),
        "referral_code": current_user.referral_code
    }
    
    # Growth Metrics
    first_transaction = db.query(TransactionHistory).filter(
        TransactionHistory.user_id == current_user.id
    ).order_by(TransactionHistory.created_at.asc()).first()
    
    days_active = (datetime.utcnow() - (first_transaction.created_at if first_transaction else current_user.created_at)).days
    
    growth_metrics = {
        "days_active": days_active,
        "average_daily_earnings": (mining_earnings + referral_earnings) / max(days_active, 1),
        "growth_rate": 0,  # Could calculate based on historical data
        "milestones": {
            "first_deposit": bool(db.query(CryptoDeposit).filter(CryptoDeposit.user_id == current_user.id).first()),
            "first_referral": len(referrals) > 0,
            "mining_active": len(active_sessions) > 0
        }
    }
    
    return UserAnalyticsResponse(
        portfolio_overview=portfolio_overview,
        mining_performance=mining_performance,
        earnings_history=earnings_history,
        transaction_analytics=transaction_analytics,
        referral_performance=referral_performance,
        growth_metrics=growth_metrics
    )

@app.get("/api/analytics/mining-performance", response_model=MiningPerformanceData)
async def get_mining_performance(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed mining performance analytics"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Daily earnings
    daily_earnings = []
    for i in range(days):
        day = datetime.utcnow() - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_mining_earnings = db.query(func.sum(TransactionHistory.amount)).filter(
            TransactionHistory.user_id == current_user.id,
            TransactionHistory.transaction_type == "mining_reward",
            TransactionHistory.created_at >= day_start,
            TransactionHistory.created_at < day_end
        ).scalar() or 0
        
        daily_earnings.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "bitcoin_earned": day_mining_earnings if day_mining_earnings else 0,
            "ethereum_earned": 0,  # Could separate by crypto type
            "total_earned": day_mining_earnings
        })
    
    # Weekly summary
    week_start = datetime.utcnow() - timedelta(days=7)
    weekly_earnings = db.query(func.sum(TransactionHistory.amount)).filter(
        TransactionHistory.user_id == current_user.id,
        TransactionHistory.transaction_type == "mining_reward",
        TransactionHistory.created_at >= week_start
    ).scalar() or 0
    
    # Monthly trends
    month_start = datetime.utcnow() - timedelta(days=30)
    monthly_earnings = db.query(func.sum(TransactionHistory.amount)).filter(
        TransactionHistory.user_id == current_user.id,
        TransactionHistory.transaction_type == "mining_reward",
        TransactionHistory.created_at >= month_start
    ).scalar() or 0
    
    # Efficiency metrics
    active_sessions = db.query(MiningSession).filter(
        MiningSession.user_id == current_user.id,
        MiningSession.is_active == True
    ).all()
    
    total_deposited = sum(session.deposited_amount for session in active_sessions)
    total_mined = sum(session.mined_amount for session in active_sessions)
    efficiency = (total_mined / total_deposited * 100) if total_deposited > 0 else 0
    
    return MiningPerformanceData(
        daily_earnings=daily_earnings,
        weekly_summary={
            "total_earned": weekly_earnings,
            "average_daily": weekly_earnings / 7,
            "best_day": max(daily_earnings[-7:], key=lambda x: x["total_earned"])["date"] if daily_earnings else None
        },
        monthly_trends={
            "total_earned": monthly_earnings,
            "trend": "up",  # Could calculate actual trend
            "growth_rate": 0
        },
        efficiency_metrics={
            "mining_efficiency": efficiency,
            "active_sessions": len(active_sessions),
            "total_mining_power": total_deposited
        }
    )

@app.get("/api/analytics/portfolio", response_model=PortfolioAnalytics)
async def get_portfolio_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio analytics and performance"""
    admin_settings = get_admin_settings(db)
    usd_values = calculate_usd_values(current_user, admin_settings)
    
    # Current value
    current_value = {
        "total_usd": usd_values["total_balance_usd"],
        "bitcoin_amount": current_user.bitcoin_balance,
        "ethereum_amount": current_user.ethereum_balance,
        "bitcoin_usd": usd_values["bitcoin_balance_usd"],
        "ethereum_usd": usd_values["ethereum_balance_usd"]
    }
    
    # Historical performance (last 30 days)
    historical_performance = []
    for i in range(30):
        day = datetime.utcnow() - timedelta(days=i)
        # This would ideally track daily balance snapshots
        # For now, we'll simulate based on transaction history
        historical_performance.append({
            "date": day.strftime("%Y-%m-%d"),
            "total_value": float(usd_values["total_balance_usd"]),  # Simplified
            "bitcoin_value": float(usd_values["bitcoin_balance_usd"]),
            "ethereum_value": float(usd_values["ethereum_balance_usd"])
        })
    
    # Asset allocation
    total_value = usd_values["total_balance_usd"]
    asset_allocation = {
        "bitcoin": {
            "percentage": (usd_values["bitcoin_balance_usd"] / total_value * 100) if total_value > 0 else 0,
            "value": usd_values["bitcoin_balance_usd"]
        },
        "ethereum": {
            "percentage": (usd_values["ethereum_balance_usd"] / total_value * 100) if total_value > 0 else 0,
            "value": usd_values["ethereum_balance_usd"]
        }
    }
    
    # Growth rate calculation
    growth_rate = {
        "daily": 0,  # Would calculate based on historical data
        "weekly": 0,
        "monthly": 0,
        "all_time": 0
    }
    
    return PortfolioAnalytics(
        current_value=current_value,
        historical_performance=historical_performance,
        asset_allocation=asset_allocation,
        growth_rate=growth_rate
    )

def log_admin_action(
    db: Session,
    admin_id: int,
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    description: Optional[str] = None,
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
        details=description,
        ip_address=ip_address,
        user_agent=user_agent,
        before_value=before_value,
        after_value=after_value
    )
    db.add(audit_log)
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

@app.get("/api/admin/audit-logs", response_model=List[AdminAuditLogResponse])
async def get_admin_audit_logs(
    limit: int = 100,
    offset: int = 0,
    action: Optional[str] = None,
    target_type: Optional[str] = None,
    admin_email: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get admin audit logs with filtering"""
    query = db.query(AdminAuditLog).options(joinedload(AdminAuditLog.admin))
    
    if action:
        query = query.filter(AdminAuditLog.action == action)
    
    if target_type:
        query = query.filter(AdminAuditLog.target_type == target_type)
    
    if admin_email:
        query = query.join(User).filter(User.email.ilike(f"%{admin_email}%"))
    
    if start_date:
        query = query.filter(AdminAuditLog.created_at >= start_date)
    
    if end_date:
        query = query.filter(AdminAuditLog.created_at <= end_date)
    
    logs = query.order_by(AdminAuditLog.created_at.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()
    
    return [
        AdminAuditLogResponse(
            id=log.id,
            admin_id=log.admin_id,
            admin_name=log.admin.name,
            admin_email=log.admin.email,
            action=log.action,
            target_type=log.target_type,
            target_id=log.target_id,
            details=log.details,
            ip_address=log.ip_address,
            created_at=log.created_at
        )
        for log in logs
    ]

@app.get("/api/admin/audit-summary")
async def get_audit_summary(
    days: int = 30,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get audit log summary and statistics"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Total actions by type
    action_counts = db.query(
        AdminAuditLog.action,
        func.count(AdminAuditLog.id).label('count')
    ).filter(
        AdminAuditLog.created_at >= start_date
    ).group_by(AdminAuditLog.action).all()
    
    # Actions by admin
    admin_counts = db.query(
        User.email,
        func.count(AdminAuditLog.id).label('count')
    ).join(AdminAuditLog).filter(
        AdminAuditLog.created_at >= start_date
    ).group_by(User.email).all()
    
    # Recent critical actions
    critical_actions = db.query(AdminAuditLog).options(joinedload(AdminAuditLog.admin)).filter(
        AdminAuditLog.action.in_(['user_suspension', 'user_flag', 'deposit_rejection', 'withdrawal_rejection']),
        AdminAuditLog.created_at >= start_date
    ).order_by(AdminAuditLog.created_at.desc()).limit(10).all()
    
    return {
        "period_days": days,
        "total_actions": sum(count for _, count in action_counts),
        "actions_by_type": {action: count for action, count in action_counts},
        "actions_by_admin": {email: count for email, count in admin_counts},
        "recent_critical_actions": [
            {
                "id": log.id,
                "admin_email": log.admin.email,
                "action": log.action,
                "target_type": log.target_type,
                "target_id": log.target_id,
                "created_at": log.created_at
            }
            for log in critical_actions
        ]
    }

@app.get("/api/user/transaction-history", response_model=List[TransactionHistoryResponse])
async def get_user_transaction_history(
    limit: int = 50,
    offset: int = 0,
    transaction_type: Optional[str] = None,
    crypto_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's transaction history with filtering"""
    query = db.query(TransactionHistory).filter(TransactionHistory.user_id == current_user.id)
    
    if transaction_type:
        query = query.filter(TransactionHistory.transaction_type == transaction_type)
    
    if crypto_type:
        query = query.filter(TransactionHistory.crypto_type == crypto_type)
    
    if start_date:
        query = query.filter(TransactionHistory.created_at >= start_date)
    
    if end_date:
        query = query.filter(TransactionHistory.created_at <= end_date)
    
    transactions = query.order_by(TransactionHistory.created_at.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()
    
    return [TransactionHistoryResponse.from_orm(t) for t in transactions]

@app.get("/api/user/transaction-summary")
async def get_user_transaction_summary(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's transaction summary statistics"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Total transactions by type
    transaction_counts = db.query(
        TransactionHistory.transaction_type,
        func.count(TransactionHistory.id).label('count'),
        func.sum(TransactionHistory.amount).label('total_amount')
    ).filter(
        TransactionHistory.user_id == current_user.id,
        TransactionHistory.created_at >= start_date
    ).group_by(TransactionHistory.transaction_type).all()
    
    # Daily transaction volume
    daily_volume = []
    for i in range(days):
        day = datetime.utcnow() - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_total = db.query(func.sum(TransactionHistory.amount)).filter(
            TransactionHistory.user_id == current_user.id,
            TransactionHistory.created_at >= day_start,
            TransactionHistory.created_at < day_end
        ).scalar() or 0
        
        daily_volume.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "volume": float(day_total)
        })
    
    return {
        "period_days": days,
        "transaction_summary": {
            t_type: {"count": count, "total_amount": float(total_amount)}
            for t_type, count, total_amount in transaction_counts
        },
        "daily_volume": daily_volume,
        "total_transactions": sum(count for _, count, _ in transaction_counts)
    }

async def send_email_notification(
    email: str,
    subject: str,
    template_type: str,
    context: dict,
    db: Session,
    attachment_path: str = None
):
    """Enhanced email notification with attachment support"""
    
    # Add deposit evidence template
    if template_type == "deposit_evidence":
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f9f9f9;">
            <div style="background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h2 style="color: #2563eb; margin-bottom: 20px;">New Deposit Evidence Received</h2>
                
                <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <h3 style="color: #374151; margin-bottom: 15px;">Deposit Details:</h3>
                    <p><strong>User:</strong> {context['user_name']} ({context['user_email']})</p>
                    <p><strong>Deposit ID:</strong> #{context['deposit_id']}</p>
                    <p><strong>Crypto Type:</strong> {context['crypto_type']}</p>
                    <p><strong>Amount:</strong> {context['amount']} {context['crypto_type']}</p>
                    <p><strong>USD Equivalent:</strong> ${context['usd_amount']}</p>
                    <p><strong>Transaction Hash:</strong> {context['transaction_hash']}</p>
                </div>
                
                <p style="color: #6b7280;">The user has uploaded evidence for this deposit. Please review and confirm or reject the deposit in the admin panel.</p>
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="#" style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">Review Deposit</a>
                </div>
            </div>
        </div>
        """
    
    elif template_type == "deposit_submission":
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f9f9f9;">
            <div style="background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h2 style="color: #2563eb; margin-bottom: 20px;">New Deposit Submission</h2>
                
                <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <h3 style="color: #374151; margin-bottom: 15px;">Deposit Details:</h3>
                    <p><strong>User:</strong> {context['user_name']} ({context['user_email']})</p>
                    <p><strong>Deposit ID:</strong> #{context['deposit_id']}</p>
                    <p><strong>Crypto Type:</strong> {context['crypto_type']}</p>
                    <p><strong>Amount:</strong> {context['amount']} {context['crypto_type']}</p>
                    <p><strong>USD Equivalent:</strong> ${context['usd_amount']}</p>
                    <p><strong>Transaction Hash:</strong> {context['transaction_hash']}</p>
                    <p><strong>Evidence Provided:</strong> {'Yes' if context['has_evidence'] else 'No'}</p>
                </div>
                
                <p style="color: #6b7280;">A user has submitted a new deposit for review. Please confirm or reject this deposit in the admin panel.</p>
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="#" style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">Review Deposit</a>
                </div>
            </div>
        </div>
        """
    
    # ... existing email templates ...
    
    # Send email with attachment if provided
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        import os
        
        msg = MIMEMultipart()
        msg['From'] = os.getenv('SMTP_FROM_EMAIL')
        msg['To'] = email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(html_content, 'html'))
        
        # Add attachment if provided
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment_path)}'
                )
                msg.attach(part)
        
        server = smtplib.SMTP(os.getenv('SMTP_HOST'), int(os.getenv('SMTP_PORT', 587)))
        server.starttls()
        server.login(os.getenv('SMTP_USERNAME'), os.getenv('SMTP_PASSWORD'))
        server.send_message(msg)
        server.quit()
        
        # Log email notification
        email_log = EmailNotification(
            email=email,
            subject=subject,
            template_type=template_type,
            status="sent",
            sent_at=datetime.utcnow()
        )
        db.add(email_log)
        db.commit()
        
    except Exception as e:
        # Log failed email
        email_log = EmailNotification(
            email=email,
            subject=subject,
            template_type=template_type,
            status="failed",
            error_message=str(e)
        )
        db.add(email_log)
        db.commit()

@app.post("/api/admin/upload-qr-code")
async def upload_qr_code(
    crypto_type: str = Form(...),
    wallet_address: str = Form(...),
    qr_file: UploadFile = File(...),
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Upload QR code image for Bitcoin or Ethereum deposits"""
    if crypto_type not in ["bitcoin", "ethereum"]:
        raise HTTPException(status_code=400, detail="Invalid crypto type. Must be 'bitcoin' or 'ethereum'")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif"]
    if qr_file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only image files (JPEG, PNG, GIF) are allowed")
    
    # Save QR code file
    import os
    import uuid
    
    upload_dir = "uploads/qr_codes"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_extension = qr_file.filename.split(".")[-1]
    unique_filename = f"{crypto_type}_qr_{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    with open(file_path, "wb") as buffer:
        content = await qr_file.read()
        buffer.write(content)
    
    qr_code_url = f"/uploads/qr_codes/{unique_filename}"
    
    # Update admin settings with new QR code and wallet address
    settings = get_or_create_admin_settings(db)
    if crypto_type == "bitcoin":
        settings.bitcoin_deposit_qr = qr_code_url
        settings.bitcoin_wallet_address = wallet_address
    else:  # ethereum
        settings.ethereum_deposit_qr = qr_code_url
        settings.ethereum_wallet_address = wallet_address
    
    db.commit()
    
    # Log admin action
    log_admin_action(
        db=db,
        admin_id=admin_user.id,
        action=f"upload_qr_code_{crypto_type}",
        target_type="admin_settings",
        target_id="1",
        details=f"Uploaded new {crypto_type} QR code with wallet address: {wallet_address}"
    )
    
    return {
        "message": f"{crypto_type.title()} QR code uploaded successfully",
        "qr_code_url": qr_code_url,
        "wallet_address": wallet_address
    }

@app.get("/api/admin/qr-codes")
def get_all_qr_codes(admin_user: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Get all QR codes for admin management"""
    qr_codes = db.query(DepositQRCode).order_by(DepositQRCode.created_at.desc()).all()
    
    return [
        {
            "id": qr.id,
            "crypto_type": qr.crypto_type,
            "qr_code_url": qr.qr_code_url,
            "wallet_address": qr.wallet_address,
            "is_active": qr.is_active,
            "created_at": qr.created_at
        }
        for qr in qr_codes
    ]

@app.delete("/api/admin/qr-codes/{qr_id}")
def delete_qr_code(
    qr_id: int,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Delete a QR code"""
    qr_code = db.query(DepositQRCode).filter(DepositQRCode.id == qr_id).first()
    if not qr_code:
        raise HTTPException(status_code=404, detail="QR code not found")
    
    # Delete file from filesystem
    import os
    if qr_code.qr_code_url.startswith("/uploads/"):
        file_path = qr_code.qr_code_url[1:]  # Remove leading slash
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Log admin action
    log_admin_action(
        db=db,
        admin_id=admin_user.id,
        action=f"delete_qr_code_{qr_code.crypto_type}",
        target_type="qr_code",
        target_id=str(qr_id),
        details=f"Deleted {qr_code.crypto_type} QR code"
    )
    
    db.delete(qr_code)
    db.commit()
    
    return {"message": "QR code deleted successfully"}

from fastapi.staticfiles import StaticFiles

# Mount static files for serving uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
