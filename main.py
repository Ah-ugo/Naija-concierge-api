from fastapi import FastAPI, Depends, HTTPException, status, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, GetJsonSchemaHandler, BeforeValidator, ValidationError
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing import List, Optional, Dict, Any, Union, Annotated
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo import MongoClient
from bson import ObjectId
import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from fastapi.encoders import jsonable_encoder
import uuid
import requests
from enum import Enum
import httpx
from pymongo.collection import Collection
import json
import hmac
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

required_env_vars = [
    "MONGO_URI",
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_API_KEY",
    "CLOUDINARY_API_SECRET",
    "SECRET_KEY",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "GMAIL_ADDRESS",
    "AIRTABLE_API_KEY",
    "AIRTABLE_BASE_ID",
    "AIRTABLE_TABLE_NAME"
]
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["naija_concierge"]

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

# Flutterwave configuration
FLUTTERWAVE_SECRET_KEY = os.getenv("FLUTTERWAVE_SECRET_KEY")
FLUTTERWAVE_BASE_URL = "https://api.flutterwave.com/v3"
FLUTTERWAVE_SECRET_HASH = os.getenv("FLUTTERWAVE_SECRET_HASH")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

app = FastAPI(title="Naija Concierge API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, handler):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_core_schema__(
            cls,
            source: type[Any],
            handler: GetJsonSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.str_schema()


# Enhanced Enums
class ServiceCategoryType(str, Enum):
    TIERED = "tiered"  # Has tiers with online payment (e.g., Sorted Lifestyle)
    CONTACT_ONLY = "contact_only"  # Individual services, contact admin (e.g., Sorted Experience, Heritage)


class BookingType(str, Enum):
    CONSULTATION = "consultation"  # Contact-only booking
    TIER_BOOKING = "tier_booking"  # Tier-based booking with payment
    MEMBERSHIP_SERVICE = "membership_service"  # Membership required


class PaymentStatus(str, Enum):
    PENDING = "pending"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MembershipTier(str, Enum):
    BASIC = "basic"
    PREMIUM = "premium"
    VIP = "vip"


class UserMembershipStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


# User Models (keeping existing structure)
class UserBase(BaseModel):
    email: EmailStr
    firstName: str
    lastName: str
    phone: Optional[str] = None
    address: Optional[str] = None
    profileImage: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    profileImage: Optional[str] = None


class UserInDB(UserBase):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    role: str = "user"
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    hashed_password: str

    model_config = {
        "populate_by_name": True,
        "json_encoders": {
            ObjectId: str
        }
    }

    @classmethod
    def model_validate(cls, obj, **kwargs):
        if isinstance(obj, dict) and '_id' in obj and isinstance(obj['_id'], ObjectId):
            obj['_id'] = str(obj['_id'])
        return super().model_validate(obj, **kwargs)


class User(UserBase):
    id: str
    role: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str
    user: User


class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None


# Service Category Models
class ServiceCategoryBase(BaseModel):
    name: str  # e.g., "Sorted Lifestyle", "Sorted Experience", "Sorted Heritage"
    description: str
    category_type: ServiceCategoryType  # TIERED or CONTACT_ONLY
    image: Optional[str] = None
    is_active: bool = True


class ServiceCategoryCreate(ServiceCategoryBase):
    pass


class ServiceCategoryUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category_type: Optional[ServiceCategoryType] = None
    image: Optional[str] = None
    is_active: Optional[bool] = None


class ServiceCategoryInDB(ServiceCategoryBase):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ServiceCategory(ServiceCategoryBase):
    id: str
    created_at: datetime
    updated_at: datetime
    tiers: List['ServiceTier'] = []  # For TIERED categories
    services: List['Service'] = []  # For CONTACT_ONLY categories

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True


# Service Tier Models (for tiered categories like Sorted Lifestyle)
class ServiceTierBase(BaseModel):
    name: str  # e.g., "Tier 1", "Tier 2"
    description: str
    price: float  # Price for this tier
    category_id: str  # Reference to ServiceCategory
    image: Optional[str] = None
    features: List[str] = []  # Tier-level features
    is_popular: bool = False
    is_available: bool = True


class ServiceTierCreate(ServiceTierBase):
    pass


class ServiceTierUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    category_id: Optional[str] = None
    image: Optional[str] = None
    features: Optional[List[str]] = None
    is_popular: Optional[bool] = None
    is_available: Optional[bool] = None


class ServiceTierInDB(ServiceTierBase):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ServiceTier(ServiceTierBase):
    id: str
    created_at: datetime
    updated_at: datetime
    services: List['Service'] = []  # Services within this tier
    category: Optional[ServiceCategory] = None

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True


# Service Models
class ServiceBase(BaseModel):
    name: str
    description: str
    image: Optional[str] = None
    duration: str
    isAvailable: bool = True
    features: List[str] = []
    requirements: List[str] = []
    # References
    category_id: Optional[str] = None  # Reference to ServiceCategory
    tier_id: Optional[str] = None  # Reference to ServiceTier (for tiered services)


class ServiceCreate(ServiceBase):
    pass


class ServiceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    image: Optional[str] = None
    duration: Optional[str] = None
    isAvailable: Optional[bool] = None
    features: Optional[List[str]] = None
    requirements: Optional[List[str]] = None
    category_id: Optional[str] = None
    tier_id: Optional[str] = None


class ServiceInDB(ServiceBase):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Service(ServiceBase):
    id: str
    createdAt: datetime
    updatedAt: datetime
    category: Optional[ServiceCategory] = None
    tier: Optional[ServiceTier] = None

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True


# Enhanced Booking Models with Payment Integration
class BookingBase(BaseModel):
    userId: str
    serviceId: Optional[str] = None  # For individual service bookings
    tierId: Optional[str] = None  # For tier bookings
    bookingDate: datetime
    status: str = "pending"
    specialRequests: Optional[str] = None
    booking_type: BookingType
    contact_preference: Optional[str] = "email"
    payment_required: bool = False
    payment_amount: Optional[float] = None
    # Enhanced payment fields
    payment_url: Optional[str] = None
    payment_status: PaymentStatus = PaymentStatus.PENDING
    payment_reference: Optional[str] = None
    flutterwave_tx_ref: Optional[str] = None


class BookingCreate(BookingBase):
    pass


class BookingUpdate(BaseModel):
    bookingDate: Optional[datetime] = None
    status: Optional[str] = None
    specialRequests: Optional[str] = None
    contact_preference: Optional[str] = None
    payment_status: Optional[PaymentStatus] = None


class BookingInDB(BookingBase):
    id: PyObjectId = Field(default_factory=ObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Booking(BookingBase):
    id: str
    createdAt: datetime
    updatedAt: datetime
    service: Optional[Service] = None
    tier: Optional[ServiceTier] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Membership Models
class MembershipBase(BaseModel):
    name: str
    description: str
    tier: MembershipTier
    price: float
    duration_months: int
    features: List[str]
    image: Optional[str] = None
    is_popular: bool = False


class MembershipCreate(MembershipBase):
    pass


class MembershipUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tier: Optional[MembershipTier] = None
    price: Optional[float] = None
    duration_months: Optional[int] = None
    features: Optional[List[str]] = None
    image: Optional[str] = None
    is_popular: Optional[bool] = None


class MembershipInDB(MembershipBase):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Membership(MembershipBase):
    id: str
    created_at: datetime
    updated_at: datetime


# User Membership Models
class UserMembershipBase(BaseModel):
    user_id: str
    membership_id: str
    start_date: datetime
    end_date: datetime
    status: UserMembershipStatus = UserMembershipStatus.ACTIVE


class UserMembershipCreate(UserMembershipBase):
    pass


class UserMembershipInDB(UserMembershipBase):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserMembership(UserMembershipBase):
    id: str
    created_at: datetime
    updated_at: datetime
    membership: Optional[Membership] = None


# Keep existing models (CRM, Package, etc.)
class CRMClientBase(BaseModel):
    clientName: str
    contactInfo: Dict[str, str]
    serviceBooked: str
    status: str = "pending"
    assignedVendor: Optional[str] = None
    notes: Optional[str] = None
    dueDate: Optional[datetime] = None


class CRMClientCreate(CRMClientBase):
    pass


class CRMClientUpdate(BaseModel):
    clientName: Optional[str] = None
    contactInfo: Optional[Dict[str, str]] = None
    serviceBooked: Optional[str] = None
    status: Optional[str] = None
    assignedVendor: Optional[str] = None
    notes: Optional[str] = None
    dueDate: Optional[datetime] = None


class CRMClientInDB(CRMClientBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CRMClient(CRMClientBase):
    id: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True


class AirtableBookingForm(BaseModel):
    clientName: str
    email: EmailStr
    phone: Optional[str] = None
    serviceId: Optional[str] = None  # For individual service booking
    tierId: Optional[str] = None  # For tier booking
    bookingDate: datetime
    specialRequests: Optional[str] = None


class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    subject: str
    message: str


class NewsletterSubscriber(BaseModel):
    email: EmailStr
    subscribed_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class NewsletterSubscriberInDB(NewsletterSubscriber):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Payment webhook models
class FlutterwaveWebhookData(BaseModel):
    id: int
    tx_ref: str
    flw_ref: str
    device_fingerprint: str
    amount: float
    currency: str
    charged_amount: float
    app_fee: float
    merchant_fee: float
    processor_response: str
    auth_model: str
    ip: str
    narration: str
    status: str
    payment_type: str
    created_at: str
    account_id: int
    customer: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None


class FlutterwaveWebhook(BaseModel):
    event: str
    data: FlutterwaveWebhookData


# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(email: str):
    user = db.users.find_one({"email": email})
    if user:
        user["_id"] = str(user["_id"])
        return UserInDB(**user)
    return None


def get_user_by_id(user_id: str):
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            logger.error(f"User not found for ID: {user_id}")
            return None
        user["_id"] = str(user["_id"])
        return UserInDB(**user)
    except ValidationError as ve:
        logger.error(f"Validation error for user ID {user_id}: {ve}")
        return None
    except Exception as e:
        logger.error(f"Error getting user by ID {user_id}: {e}", exc_info=True)
        return None


def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = get_user(email=token_data.email)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    return current_user


async def get_admin_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


def check_user_membership(user_id: str) -> bool:
    """Helper function to check if user has active membership"""
    active_membership = db.user_memberships.find_one({
        "user_id": user_id,
        "status": UserMembershipStatus.ACTIVE,
        "end_date": {"$gt": datetime.utcnow()}
    })
    return active_membership is not None


def send_email(to_email: str, subject: str, html_content: str):
    """Send email using SMTP, with improved connection handling."""
    smtp_server = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("GMAIL_ADDRESS")

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to_email

    html_part = MIMEText(html_content, "html")
    message.attach(html_part)

    server = None
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, to_email, message.as_string())
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False
    finally:
        if server:
            try:
                server.quit()
            except Exception as e:
                logger.error(f"Error closing SMTP connection: {e}")


def send_admin_notification(subject: str, message: str):
    """Send an email notification to all admin users."""
    try:
        admin_users = db.users.find({"role": "admin"})
        admin_emails = [user["email"] for user in admin_users]

        if not admin_emails:
            logger.warning("No admin users found for notification")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No admin users found to notify"
            )

        for email in admin_emails:
            try:
                send_email(email, subject, message)
                logger.info(f"Notification sent to admin: {email}")
            except Exception as e:
                logger.error(f"Failed to send notification to {email}: {e}")
                continue

        logger.info(f"Notifications sent to {len(admin_emails)} admins")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending admin notifications: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send admin notifications"
        )


def upload_file_to_cloudinary(file: UploadFile, folder: str = "naija_concierge"):
    """Upload file to Cloudinary and return the URL"""
    try:
        result = cloudinary.uploader.upload(
            file.file,
            folder=folder,
            public_id=f"{uuid.uuid4()}",
            overwrite=True
        )
        return result.get("secure_url")
    except Exception as e:
        logger.error(f"Error uploading to Cloudinary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file"
        )


def add_to_airtable(booking_data: Dict) -> Dict:
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('AIRTABLE_API_KEY')}",
            "Content-Type": "application/json"
        }
        url = f"https://api.airtable.com/v0/{os.getenv('AIRTABLE_BASE_ID')}/{os.getenv('AIRTABLE_TABLE_NAME')}"

        serialized_data = {
            "records": [{
                "fields": {
                    key: str(value) if isinstance(value, (ObjectId, datetime)) else value
                    for key, value in booking_data.items()
                }
            }]
        }
        logger.info(f"Sending to Airtable: {serialized_data}")

        response = requests.post(url, headers=headers, json=serialized_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json() if e.response.content else str(e)
        logger.error(f"Airtable HTTP error: {e}, Response: {error_detail}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Airtable API error: {error_detail}"
        )
    except Exception as e:
        logger.error(f"Error adding to Airtable: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add to Airtable: {str(e)}"
        )


# Flutterwave Payment Integration Functions
def generate_flutterwave_payment_url(booking_data: Dict, user_data: Dict, amount: float) -> str:
    """Generate Flutterwave payment URL for tier bookings"""
    if not FLUTTERWAVE_SECRET_KEY:
        logger.error("Flutterwave secret key not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Payment service not configured"
        )

    try:
        # Generate unique transaction reference
        tx_ref = f"booking_{booking_data['id']}_{uuid.uuid4().hex[:8]}"

        # Flutterwave payment payload
        payment_payload = {
            "tx_ref": tx_ref,
            "amount": amount,
            "currency": "NGN",
            "redirect_url": f"{FRONTEND_URL}/booking/payment-success",
            "payment_options": "card,banktransfer,ussd",
            "customer": {
                "email": user_data["email"],
                "phonenumber": user_data.get("phone", ""),
                "name": f"{user_data['firstName']} {user_data['lastName']}"
            },
            "customizations": {
                "title": "Naija Concierge - Tier Booking",
                "description": f"Payment for tier booking",
                "logo": "https://your-logo-url.com/logo.png"
            },
            "meta": {
                "booking_id": str(booking_data["id"]),
                "user_id": user_data["id"],
                "booking_type": "tier_booking"
            }
        }

        headers = {
            "Authorization": f"Bearer {FLUTTERWAVE_SECRET_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{FLUTTERWAVE_BASE_URL}/payments",
            json=payment_payload,
            headers=headers
        )
        response.raise_for_status()

        payment_data = response.json()
        if payment_data["status"] == "success":
            # Update booking with transaction reference
            db.bookings.update_one(
                {"_id": ObjectId(booking_data["id"])},
                {"$set": {"flutterwave_tx_ref": tx_ref}}
            )
            return payment_data["data"]["link"]
        else:
            logger.error(f"Payment URL generation failed: {payment_data}")
            raise Exception(f"Payment URL generation failed: {payment_data}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Flutterwave API request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate payment URL"
        )
    except Exception as e:
        logger.error(f"Flutterwave payment URL generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate payment URL"
        )


def verify_flutterwave_payment(tx_ref: str) -> Dict:
    """Verify payment with Flutterwave"""
    if not FLUTTERWAVE_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Payment service not configured"
        )

    try:
        headers = {
            "Authorization": f"Bearer {FLUTTERWAVE_SECRET_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.get(
            f"{FLUTTERWAVE_BASE_URL}/transactions/verify_by_reference?tx_ref={tx_ref}",
            headers=headers
        )
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Flutterwave verification request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify payment"
        )


def verify_webhook_signature(payload: str, signature: str) -> bool:
    """Verify Flutterwave webhook signature"""
    if not FLUTTERWAVE_SECRET_HASH:
        logger.warning("Flutterwave secret hash not configured")
        return True  # Skip verification if not configured

    try:
        expected_signature = hmac.new(
            FLUTTERWAVE_SECRET_HASH.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected_signature, signature)
    except Exception as e:
        logger.error(f"Error verifying webhook signature: {e}")
        return False


# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Naija Concierge API"}


# Auth routes (keeping existing)
@app.post("/auth/register", response_model=Token)
async def register(user: UserCreate):
    if db.users.find_one({"email": user.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    del user_dict["password"]
    user_in_db = UserInDB(
        **user_dict,
        hashed_password=hashed_password,
        role="user"
    )

    result = db.users.insert_one(user_in_db.dict(by_alias=True))

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    created_user = db.users.find_one({"_id": result.inserted_id})
    user_response = User(
        id=str(created_user["_id"]),
        email=created_user["email"],
        firstName=created_user["firstName"],
        lastName=created_user["lastName"],
        phone=created_user.get("phone"),
        address=created_user.get("address"),
        profileImage=created_user.get("profileImage"),
        role=created_user["role"],
        createdAt=created_user["createdAt"],
        updatedAt=created_user["updatedAt"]
    )

    welcome_html = f"""
    <html>
        <body>
            <h1>Welcome to Naija Concierge, {user.firstName}!</h1>
            <p>Thank you for registering with us. We're excited to help you experience Lagos like never before.</p>
            <p>If you have any questions or need assistance, please don't hesitate to contact us.</p>
            <p>Best regards,<br>The Naija Concierge Team</p>
        </body>
    </html>
    """
    send_email(user.email, "Welcome to Naija Concierge", welcome_html)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }


@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    user_response = User(
        id=str(user.id),
        email=user.email,
        firstName=user.firstName,
        lastName=user.lastName,
        phone=user.phone,
        address=user.address,
        profileImage=user.profileImage,
        role=user.role,
        createdAt=user.createdAt,
        updatedAt=user.updatedAt
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }


@app.get("/auth/me", response_model=User)
async def get_me(current_user: UserInDB = Depends(get_current_active_user)):
    return User(
        id=str(current_user.id),
        email=current_user.email,
        firstName=current_user.firstName,
        lastName=current_user.lastName,
        phone=current_user.phone,
        address=current_user.address,
        profileImage=current_user.profileImage,
        role=current_user.role,
        createdAt=current_user.createdAt,
        updatedAt=current_user.updatedAt
    )


# Service Category routes
@app.get("/service-categories", response_model=List[ServiceCategory])
async def get_service_categories(
        skip: int = 0,
        limit: int = 100,
        category_type: Optional[ServiceCategoryType] = None
):
    """Get all service categories with their tiers/services"""
    query = {}
    if category_type:
        query["category_type"] = category_type

    categories = list(db.service_categories.find(query).skip(skip).limit(limit))
    result = []

    for category in categories:
        tiers = []
        services = []

        if category["category_type"] == ServiceCategoryType.TIERED:
            # Get tiers for this category
            tier_docs = list(db.service_tiers.find({"category_id": str(category["_id"])}))
            for tier_doc in tier_docs:
                # Get services for this tier
                tier_services = list(db.services.find({"tier_id": str(tier_doc["_id"])}))
                tier_service_objects = []

                for service in tier_services:
                    tier_service_objects.append(
                        Service(
                            id=str(service["_id"]),
                            name=service["name"],
                            description=service["description"],
                            image=service.get("image"),
                            duration=service["duration"],
                            isAvailable=service["isAvailable"],
                            features=service.get("features", []),
                            requirements=service.get("requirements", []),
                            category_id=service.get("category_id"),
                            tier_id=service.get("tier_id"),
                            createdAt=service["createdAt"],
                            updatedAt=service["updatedAt"]
                        )
                    )

                tiers.append(
                    ServiceTier(
                        id=str(tier_doc["_id"]),
                        name=tier_doc["name"],
                        description=tier_doc["description"],
                        price=tier_doc["price"],
                        category_id=tier_doc["category_id"],
                        image=tier_doc.get("image"),
                        features=tier_doc.get("features", []),
                        is_popular=tier_doc.get("is_popular", False),
                        is_available=tier_doc.get("is_available", True),
                        created_at=tier_doc["created_at"],
                        updated_at=tier_doc["updated_at"],
                        services=tier_service_objects
                    )
                )
        else:
            # Get individual services for this category
            service_docs = list(db.services.find({"category_id": str(category["_id"]), "tier_id": None}))
            for service in service_docs:
                services.append(
                    Service(
                        id=str(service["_id"]),
                        name=service["name"],
                        description=service["description"],
                        image=service.get("image"),
                        duration=service["duration"],
                        isAvailable=service["isAvailable"],
                        features=service.get("features", []),
                        requirements=service.get("requirements", []),
                        category_id=service.get("category_id"),
                        tier_id=service.get("tier_id"),
                        createdAt=service["createdAt"],
                        updatedAt=service["updatedAt"]
                    )
                )

        result.append(
            ServiceCategory(
                id=str(category["_id"]),
                name=category["name"],
                description=category["description"],
                category_type=category["category_type"],
                image=category.get("image"),
                is_active=category["is_active"],
                created_at=category["created_at"],
                updated_at=category["updated_at"],
                tiers=tiers,
                services=services
            )
        )

    return result


@app.get("/service-categories/{category_id}", response_model=ServiceCategory)
async def get_service_category(category_id: str):
    """Get a specific service category with its tiers/services"""
    try:
        category = db.service_categories.find_one({"_id": ObjectId(category_id)})
        if not category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service category not found"
            )

        tiers = []
        services = []

        if category["category_type"] == ServiceCategoryType.TIERED:
            # Get tiers for this category
            tier_docs = list(db.service_tiers.find({"category_id": category_id}))
            for tier_doc in tier_docs:
                # Get services for this tier
                tier_services = list(db.services.find({"tier_id": str(tier_doc["_id"])}))
                tier_service_objects = []

                for service in tier_services:
                    tier_service_objects.append(
                        Service(
                            id=str(service["_id"]),
                            name=service["name"],
                            description=service["description"],
                            image=service.get("image"),
                            duration=service["duration"],
                            isAvailable=service["isAvailable"],
                            features=service.get("features", []),
                            requirements=service.get("requirements", []),
                            category_id=service.get("category_id"),
                            tier_id=service.get("tier_id"),
                            createdAt=service["createdAt"],
                            updatedAt=service["updatedAt"]
                        )
                    )

                tiers.append(
                    ServiceTier(
                        id=str(tier_doc["_id"]),
                        name=tier_doc["name"],
                        description=tier_doc["description"],
                        price=tier_doc["price"],
                        category_id=tier_doc["category_id"],
                        image=tier_doc.get("image"),
                        features=tier_doc.get("features", []),
                        is_popular=tier_doc.get("is_popular", False),
                        is_available=tier_doc.get("is_available", True),
                        created_at=tier_doc["created_at"],
                        updated_at=tier_doc["updated_at"],
                        services=tier_service_objects
                    )
                )
        else:
            # Get individual services for this category
            service_docs = list(db.services.find({"category_id": category_id, "tier_id": None}))
            for service in service_docs:
                services.append(
                    Service(
                        id=str(service["_id"]),
                        name=service["name"],
                        description=service["description"],
                        image=service.get("image"),
                        duration=service["duration"],
                        isAvailable=service["isAvailable"],
                        features=service.get("features", []),
                        requirements=service.get("requirements", []),
                        category_id=service.get("category_id"),
                        tier_id=service.get("tier_id"),
                        createdAt=service["createdAt"],
                        updatedAt=service["updatedAt"]
                    )
                )

        return ServiceCategory(
            id=str(category["_id"]),
            name=category["name"],
            description=category["description"],
            category_type=category["category_type"],
            image=category.get("image"),
            is_active=category["is_active"],
            created_at=category["created_at"],
            updated_at=category["updated_at"],
            tiers=tiers,
            services=services
        )
    except Exception as e:
        logger.error(f"Error getting service category: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service category not found"
        )


@app.post("/service-categories", response_model=ServiceCategory)
async def create_service_category(
        category: ServiceCategoryCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Create a new service category"""
    if not category.name.strip() or not category.description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name and description are required"
        )

    category_in_db = ServiceCategoryInDB(**category.dict())
    category_dict = category_in_db.dict(by_alias=True)
    if category_dict.get("_id") is None:
        del category_dict["_id"]

    result = db.service_categories.insert_one(category_dict)
    created_category = db.service_categories.find_one({"_id": result.inserted_id})

    return ServiceCategory(
        id=str(created_category["_id"]),
        name=created_category["name"],
        description=created_category["description"],
        category_type=created_category["category_type"],
        image=created_category.get("image"),
        is_active=created_category["is_active"],
        created_at=created_category["created_at"],
        updated_at=created_category["updated_at"],
        tiers=[],
        services=[]
    )


@app.post("/service-categories/image")
async def upload_service_category_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_admin_user)
):
    """Upload service category image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/categories")
    return {"imageUrl": image_url}


# Service Tier routes (for tiered categories)
@app.get("/service-tiers", response_model=List[ServiceTier])
async def get_service_tiers(
        skip: int = 0,
        limit: int = 100,
        category_id: Optional[str] = None
):
    """Get service tiers with their services"""
    query = {}
    if category_id:
        query["category_id"] = category_id

    tiers = list(db.service_tiers.find(query).skip(skip).limit(limit))
    result = []

    for tier in tiers:
        # Get services for this tier
        services = list(db.services.find({"tier_id": str(tier["_id"])}))
        service_objects = []

        for service in services:
            service_objects.append(
                Service(
                    id=str(service["_id"]),
                    name=service["name"],
                    description=service["description"],
                    image=service.get("image"),
                    duration=service["duration"],
                    isAvailable=service["isAvailable"],
                    features=service.get("features", []),
                    requirements=service.get("requirements", []),
                    category_id=service.get("category_id"),
                    tier_id=service.get("tier_id"),
                    createdAt=service["createdAt"],
                    updatedAt=service["updatedAt"]
                )
            )

        # Get category info
        category_obj = None
        if tier.get("category_id"):
            category = db.service_categories.find_one({"_id": ObjectId(tier["category_id"])})
            if category:
                category_obj = ServiceCategory(
                    id=str(category["_id"]),
                    name=category["name"],
                    description=category["description"],
                    category_type=category["category_type"],
                    image=category.get("image"),
                    is_active=category["is_active"],
                    created_at=category["created_at"],
                    updated_at=category["updated_at"],
                    tiers=[],
                    services=[]
                )

        result.append(
            ServiceTier(
                id=str(tier["_id"]),
                name=tier["name"],
                description=tier["description"],
                price=tier["price"],
                category_id=tier["category_id"],
                image=tier.get("image"),
                features=tier.get("features", []),
                is_popular=tier.get("is_popular", False),
                is_available=tier.get("is_available", True),
                created_at=tier["created_at"],
                updated_at=tier["updated_at"],
                services=service_objects,
                category=category_obj
            )
        )

    return result


@app.get("/service-tiers/{tier_id}", response_model=ServiceTier)
async def get_service_tier(tier_id: str):
    """Get a specific service tier with its services"""
    try:
        tier = db.service_tiers.find_one({"_id": ObjectId(tier_id)})
        if not tier:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service tier not found"
            )

        # Get services for this tier
        services = list(db.services.find({"tier_id": tier_id}))
        service_objects = []

        for service in services:
            service_objects.append(
                Service(
                    id=str(service["_id"]),
                    name=service["name"],
                    description=service["description"],
                    image=service.get("image"),
                    duration=service["duration"],
                    isAvailable=service["isAvailable"],
                    features=service.get("features", []),
                    requirements=service.get("requirements", []),
                    category_id=service.get("category_id"),
                    tier_id=service.get("tier_id"),
                    createdAt=service["createdAt"],
                    updatedAt=service["updatedAt"]
                )
            )

        # Get category info
        category_obj = None
        if tier.get("category_id"):
            category = db.service_categories.find_one({"_id": ObjectId(tier["category_id"])})
            if category:
                category_obj = ServiceCategory(
                    id=str(category["_id"]),
                    name=category["name"],
                    description=category["description"],
                    category_type=category["category_type"],
                    image=category.get("image"),
                    is_active=category["is_active"],
                    created_at=category["created_at"],
                    updated_at=category["updated_at"],
                    tiers=[],
                    services=[]
                )

        return ServiceTier(
            id=str(tier["_id"]),
            name=tier["name"],
            description=tier["description"],
            price=tier["price"],
            category_id=tier["category_id"],
            image=tier.get("image"),
            features=tier.get("features", []),
            is_popular=tier.get("is_popular", False),
            is_available=tier.get("is_available", True),
            created_at=tier["created_at"],
            updated_at=tier["updated_at"],
            services=service_objects,
            category=category_obj
        )
    except Exception as e:
        logger.error(f"Error getting service tier: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service tier not found"
        )


@app.post("/service-tiers", response_model=ServiceTier)
async def create_service_tier(
        tier: ServiceTierCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Create a new service tier"""
    if not tier.name.strip() or not tier.description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name and description are required"
        )

    if tier.price < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Price cannot be negative"
        )

    # Validate category exists and is tiered
    category = db.service_categories.find_one({"_id": ObjectId(tier.category_id)})
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service category not found"
        )

    if category["category_type"] != ServiceCategoryType.TIERED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only create tiers for tiered categories"
        )

    tier_in_db = ServiceTierInDB(**tier.dict())
    tier_dict = tier_in_db.dict(by_alias=True)
    if tier_dict.get("_id") is None:
        del tier_dict["_id"]

    result = db.service_tiers.insert_one(tier_dict)
    created_tier = db.service_tiers.find_one({"_id": result.inserted_id})

    return ServiceTier(
        id=str(created_tier["_id"]),
        name=created_tier["name"],
        description=created_tier["description"],
        price=created_tier["price"],
        category_id=created_tier["category_id"],
        image=created_tier.get("image"),
        features=created_tier.get("features", []),
        is_popular=created_tier.get("is_popular", False),
        is_available=created_tier.get("is_available", True),
        created_at=created_tier["created_at"],
        updated_at=created_tier["updated_at"],
        services=[]
    )


@app.post("/service-tiers/image")
async def upload_service_tier_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_admin_user)
):
    """Upload service tier image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/service_tiers")
    return {"imageUrl": image_url}


# Service routes
@app.get("/services", response_model=List[Service])
async def get_services(
        skip: int = 0,
        limit: int = 100,
        category_id: Optional[str] = None,
        tier_id: Optional[str] = None
):
    """Get services"""
    query = {}

    if category_id:
        query["category_id"] = category_id
    if tier_id:
        query["tier_id"] = tier_id

    services = list(db.services.find(query).skip(skip).limit(limit))
    result = []

    for service in services:
        # Get category info
        category_obj = None
        if service.get("category_id"):
            category = db.service_categories.find_one({"_id": ObjectId(service["category_id"])})
            if category:
                category_obj = ServiceCategory(
                    id=str(category["_id"]),
                    name=category["name"],
                    description=category["description"],
                    category_type=category["category_type"],
                    image=category.get("image"),
                    is_active=category["is_active"],
                    created_at=category["created_at"],
                    updated_at=category["updated_at"],
                    tiers=[],
                    services=[]
                )

        # Get tier info
        tier_obj = None
        if service.get("tier_id"):
            tier = db.service_tiers.find_one({"_id": ObjectId(service["tier_id"])})
            if tier:
                tier_obj = ServiceTier(
                    id=str(tier["_id"]),
                    name=tier["name"],
                    description=tier["description"],
                    price=tier["price"],
                    category_id=tier["category_id"],
                    image=tier.get("image"),
                    features=tier.get("features", []),
                    is_popular=tier.get("is_popular", False),
                    is_available=tier.get("is_available", True),
                    created_at=tier["created_at"],
                    updated_at=tier["updated_at"],
                    services=[]
                )

        result.append(
            Service(
                id=str(service["_id"]),
                name=service["name"],
                description=service["description"],
                image=service.get("image"),
                duration=service["duration"],
                isAvailable=service["isAvailable"],
                features=service.get("features", []),
                requirements=service.get("requirements", []),
                category_id=service.get("category_id"),
                tier_id=service.get("tier_id"),
                createdAt=service["createdAt"],
                updatedAt=service["updatedAt"],
                category=category_obj,
                tier=tier_obj
            )
        )

    return result


@app.get("/services/{service_id}", response_model=Service)
async def get_service(service_id: str):
    """Get a specific service"""
    try:
        service = db.services.find_one({"_id": ObjectId(service_id)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found"
            )

        # Get category info
        category_obj = None
        if service.get("category_id"):
            category = db.service_categories.find_one({"_id": ObjectId(service["category_id"])})
            if category:
                category_obj = ServiceCategory(
                    id=str(category["_id"]),
                    name=category["name"],
                    description=category["description"],
                    category_type=category["category_type"],
                    image=category.get("image"),
                    is_active=category["is_active"],
                    created_at=category["created_at"],
                    updated_at=category["updated_at"],
                    tiers=[],
                    services=[]
                )

        # Get tier info
        tier_obj = None
        if service.get("tier_id"):
            tier = db.service_tiers.find_one({"_id": ObjectId(service["tier_id"])})
            if tier:
                tier_obj = ServiceTier(
                    id=str(tier["_id"]),
                    name=tier["name"],
                    description=tier["description"],
                    price=tier["price"],
                    category_id=tier["category_id"],
                    image=tier.get("image"),
                    features=tier.get("features", []),
                    is_popular=tier.get("is_popular", False),
                    is_available=tier.get("is_available", True),
                    created_at=tier["created_at"],
                    updated_at=tier["updated_at"],
                    services=[]
                )

        return Service(
            id=str(service["_id"]),
            name=service["name"],
            description=service["description"],
            image=service.get("image"),
            duration=service["duration"],
            isAvailable=service["isAvailable"],
            features=service.get("features", []),
            requirements=service.get("requirements", []),
            category_id=service.get("category_id"),
            tier_id=service.get("tier_id"),
            createdAt=service["createdAt"],
            updatedAt=service["updatedAt"],
            category=category_obj,
            tier=tier_obj
        )
    except Exception as e:
        logger.error(f"Error getting service: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )


@app.post("/services", response_model=Service)
async def create_service(
        service: ServiceCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Create a new service"""
    if not service.name.strip() or not service.description.strip() or not service.duration.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name, description, and duration are required"
        )

    # Validate category exists
    if service.category_id:
        category = db.service_categories.find_one({"_id": ObjectId(service.category_id)})
        if not category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service category not found"
            )

    # Validate tier if specified
    if service.tier_id:
        tier = db.service_tiers.find_one({"_id": ObjectId(service.tier_id)})
        if not tier:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service tier not found"
            )

        # Ensure tier belongs to the specified category
        if service.category_id and tier["category_id"] != service.category_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tier does not belong to the specified category"
            )

    service_in_db = ServiceInDB(**service.dict())
    service_dict = service_in_db.dict(by_alias=True)
    if service_dict.get("_id") is None:
        del service_dict["_id"]
    result = db.services.insert_one(service_dict)

    created_service = db.services.find_one({"_id": result.inserted_id})

    # Get category and tier info
    category_obj = None
    tier_obj = None

    if created_service.get("category_id"):
        category = db.service_categories.find_one({"_id": ObjectId(created_service["category_id"])})
        if category:
            category_obj = ServiceCategory(
                id=str(category["_id"]),
                name=category["name"],
                description=category["description"],
                category_type=category["category_type"],
                image=category.get("image"),
                is_active=category["is_active"],
                created_at=category["created_at"],
                updated_at=category["updated_at"],
                tiers=[],
                services=[]
            )

    if created_service.get("tier_id"):
        tier = db.service_tiers.find_one({"_id": ObjectId(created_service["tier_id"])})
        if tier:
            tier_obj = ServiceTier(
                id=str(tier["_id"]),
                name=tier["name"],
                description=tier["description"],
                price=tier["price"],
                category_id=tier["category_id"],
                image=tier.get("image"),
                features=tier.get("features", []),
                is_popular=tier.get("is_popular", False),
                is_available=tier.get("is_available", True),
                created_at=tier["created_at"],
                updated_at=tier["updated_at"],
                services=[]
            )

    return Service(
        id=str(created_service["_id"]),
        name=created_service["name"],
        description=created_service["description"],
        image=created_service.get("image"),
        duration=created_service["duration"],
        isAvailable=created_service["isAvailable"],
        features=created_service.get("features", []),
        requirements=created_service.get("requirements", []),
        category_id=created_service.get("category_id"),
        tier_id=created_service.get("tier_id"),
        createdAt=created_service["createdAt"],
        updatedAt=created_service["updatedAt"],
        category=category_obj,
        tier=tier_obj
    )


@app.post("/services/image")
async def upload_service_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_admin_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/services")
    return {"imageUrl": image_url}


@app.post("/bookings", response_model=Booking)
async def create_booking(
        booking: BookingCreate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Create a new booking for either a service category or tier with payment integration"""
    if current_user.role != "admin" and booking.userId != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Must have either serviceId (category) or tierId, but not both
    if not booking.serviceId and not booking.tierId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either serviceId (category) or tierId must be provided"
        )

    if booking.serviceId and booking.tierId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot book both service category and tier simultaneously"
        )

    service_obj = None
    tier_obj = None
    category_obj = None

    if booking.serviceId:
        # Individual service category booking (contact-only)
        try:
            # Query service_categories collection instead of services
            category = db.service_categories.find_one({"_id": ObjectId(booking.serviceId)})
            if not category:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Service category not found"
                )

            if not category["is_active"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Service category is not available"
                )

            # Verify it's a contact-only category
            if category["category_type"] != ServiceCategoryType.CONTACT_ONLY:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Individual service booking only allowed for contact-only categories"
                )

            booking.booking_type = BookingType.CONSULTATION
            booking.payment_required = False
            booking.payment_amount = 0.0
            booking.payment_status = PaymentStatus.PENDING

            # Get services within this category
            category_services = list(db.services.find({"category_id": booking.serviceId}))
            category_service_objects = []

            for service in category_services:
                category_service_objects.append(
                    Service(
                        id=str(service["_id"]),
                        name=service["name"],
                        description=service["description"],
                        image=service.get("image"),
                        duration=service["duration"],
                        isAvailable=service["isAvailable"],
                        features=service.get("features", []),
                        requirements=service.get("requirements", []),
                        category_id=service.get("category_id"),
                        tier_id=service.get("tier_id"),
                        createdAt=service["createdAt"],
                        updatedAt=service["updatedAt"]
                    )
                )

            category_obj = ServiceCategory(
                id=str(category["_id"]),
                name=category["name"],
                description=category["description"],
                category_type=category["category_type"],
                image=category.get("image"),
                is_active=category["is_active"],
                created_at=category["created_at"],
                updated_at=category["updated_at"],
                tiers=[],
                services=category_service_objects
            )

        except Exception as e:
            logger.error(f"Error checking service category: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid service category ID"
            )

    elif booking.tierId:
        # Tier booking (online payment)
        try:
            # Query service_tiers collection
            tier = db.service_tiers.find_one({"_id": ObjectId(booking.tierId)})
            if not tier:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Service tier not found"
                )

            if not tier["is_available"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Service tier is not available"
                )

            # Get the parent category
            category = db.service_categories.find_one({"_id": ObjectId(tier["category_id"])})
            if not category:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Service category not found"
                )

            # Verify it's a tiered category
            if category["category_type"] != ServiceCategoryType.TIERED:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tier booking only allowed for tiered categories"
                )

            booking.booking_type = BookingType.TIER_BOOKING
            booking.payment_required = True
            booking.payment_amount = tier["price"]
            booking.payment_status = PaymentStatus.PENDING

            # Get services in this tier
            tier_services = list(db.services.find({"tier_id": booking.tierId}))
            tier_service_objects = []

            for service in tier_services:
                tier_service_objects.append(
                    Service(
                        id=str(service["_id"]),
                        name=service["name"],
                        description=service["description"],
                        image=service.get("image"),
                        duration=service["duration"],
                        isAvailable=service["isAvailable"],
                        features=service.get("features", []),
                        requirements=service.get("requirements", []),
                        category_id=service.get("category_id"),
                        tier_id=service.get("tier_id"),
                        createdAt=service["createdAt"],
                        updatedAt=service["updatedAt"]
                    )
                )

            tier_obj = ServiceTier(
                id=str(tier["_id"]),
                name=tier["name"],
                description=tier["description"],
                price=tier["price"],
                category_id=tier["category_id"],
                image=tier.get("image"),
                features=tier.get("features", []),
                is_popular=tier.get("is_popular", False),
                is_available=tier.get("is_available", True),
                created_at=tier["created_at"],
                updated_at=tier["updated_at"],
                services=tier_service_objects
            )

            # Get all tiers for this category
            category_tiers = list(db.service_tiers.find({"category_id": tier["category_id"]}))
            category_tier_objects = []

            for cat_tier in category_tiers:
                # Get services for each tier
                tier_services_for_cat = list(db.services.find({"tier_id": str(cat_tier["_id"])}))
                tier_service_objects_for_cat = []

                for service in tier_services_for_cat:
                    tier_service_objects_for_cat.append(
                        Service(
                            id=str(service["_id"]),
                            name=service["name"],
                            description=service["description"],
                            image=service.get("image"),
                            duration=service["duration"],
                            isAvailable=service["isAvailable"],
                            features=service.get("features", []),
                            requirements=service.get("requirements", []),
                            category_id=service.get("category_id"),
                            tier_id=service.get("tier_id"),
                            createdAt=service["createdAt"],
                            updatedAt=service["updatedAt"]
                        )
                    )

                category_tier_objects.append(
                    ServiceTier(
                        id=str(cat_tier["_id"]),
                        name=cat_tier["name"],
                        description=cat_tier["description"],
                        price=cat_tier["price"],
                        category_id=cat_tier["category_id"],
                        image=cat_tier.get("image"),
                        features=cat_tier.get("features", []),
                        is_popular=cat_tier.get("is_popular", False),
                        is_available=cat_tier.get("is_available", True),
                        created_at=cat_tier["created_at"],
                        updated_at=cat_tier["updated_at"],
                        services=tier_service_objects_for_cat
                    )
                )

            category_obj = ServiceCategory(
                id=str(category["_id"]),
                name=category["name"],
                description=category["description"],
                category_type=category["category_type"],
                image=category.get("image"),
                is_active=category["is_active"],
                created_at=category["created_at"],
                updated_at=category["updated_at"],
                tiers=category_tier_objects,
                services=[]
            )

        except Exception as e:
            logger.error(f"Error checking tier: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tier ID"
            )

    # Create booking in database
    booking_in_db = BookingInDB(**booking.dict())
    result = db.bookings.insert_one(booking_in_db.dict(by_alias=True))
    created_booking = db.bookings.find_one({"_id": result.inserted_id})

    # Generate payment URL for tier bookings
    payment_url = None
    if booking.booking_type == BookingType.TIER_BOOKING and booking.payment_required:
        try:
            user = get_user_by_id(booking.userId)
            if user:
                payment_url = generate_flutterwave_payment_url(
                    {"id": str(created_booking["_id"])},
                    user.__dict__,
                    booking.payment_amount
                )

                # Update booking with payment URL
                db.bookings.update_one(
                    {"_id": result.inserted_id},
                    {"$set": {"payment_url": payment_url}}
                )

                # Update the created_booking dict for response
                created_booking["payment_url"] = payment_url

        except Exception as e:
            logger.error(f"Payment URL generation failed: {e}")
            # Continue without payment URL - can be generated later

    # Send emails and notifications
    user = get_user_by_id(booking.userId)
    if user:
        if booking.booking_type == BookingType.CONSULTATION:
            booking_html = f"""
            <html>
                <body>
                    <h1>Service Request Received</h1>
                    <p>Dear {user.firstName},</p>
                    <p>We have received your request for services from {category_obj.name}.</p>
                    <p>Our team will contact you within 24 hours to discuss your requirements and provide a custom quote.</p>
                    <p>Booking Details:</p>
                    <ul>
                        <li>Category: {category_obj.name}</li>
                        <li>Available Services: {', '.join([s.name for s in category_obj.services])}</li>
                        <li>Preferred Date: {created_booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}</li>
                        <li>Contact Preference: {created_booking.get("contact_preference", "email")}</li>
                        <li>Special Requests: {created_booking.get("specialRequests", "None")}</li>
                    </ul>
                    <p>Best regards,<br>The Naija Concierge Team</p>
                </body>
            </html>
            """
            send_email(user.email, "Service Request - Naija Concierge", booking_html)

        elif booking.booking_type == BookingType.TIER_BOOKING:
            service_list = ", ".join([s.name for s in tier_obj.services])
            payment_info = f"<p>Payment URL: <a href='{payment_url}'>Complete Payment</a></p>" if payment_url else "<p>Payment URL will be provided shortly.</p>"
            booking_html = f"""
            <html>
                <body>
                    <h1>Tier Booking Confirmation - Payment Required</h1>
                    <p>Dear {user.firstName},</p>
                    <p>Your booking for {tier_obj.name} from {category_obj.name} has been received.</p>
                    <p>To confirm your booking, please complete the payment of ₦{tier_obj.price:,.2f}.</p>
                    {payment_info}
                    <p>Booking Details:</p>
                    <ul>
                        <li>Tier: {tier_obj.name}</li>
                        <li>Category: {category_obj.name}</li>
                        <li>Included Services: {service_list}</li>
                        <li>Date: {created_booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}</li>
                        <li>Amount: ₦{tier_obj.price:,.2f}</li>
                        <li>Status: Pending Payment</li>
                    </ul>
                    <p>Best regards,<br>The Naija Concierge Team</p>
                </body>
            </html>
            """
            send_email(user.email, "Tier Booking Confirmation - Payment Required", booking_html)

    # Send admin notification
    if booking.booking_type == BookingType.CONSULTATION:
        available_services = ", ".join([s.name for s in category_obj.services])
        notification_message = f"""
        New service category request:
        - Client: {user.firstName} {user.lastName}
        - Category: {category_obj.name}
        - Available Services: {available_services}
        - Date: {created_booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}
        - Type: Contact Required
        - Special Requests: {created_booking.get("specialRequests", "None")}
        """
    else:
        service_list = ", ".join([s.name for s in tier_obj.services])
        notification_message = f"""
        New tier booking:
        - Client: {user.firstName} {user.lastName}
        - Tier: {tier_obj.name}
        - Category: {category_obj.name}
        - Services: {service_list}
        - Date: {created_booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}
        - Amount: ₦{tier_obj.price:,.2f}
        - Payment Required: Yes
        - Payment URL: {payment_url or "Generation failed"}
        """

    send_admin_notification("New Booking Created", notification_message)

    return Booking(
        id=str(created_booking["_id"]),
        userId=created_booking["userId"],
        serviceId=created_booking.get("serviceId"),
        tierId=created_booking.get("tierId"),
        bookingDate=created_booking["bookingDate"],
        status=created_booking["status"],
        specialRequests=created_booking.get("specialRequests"),
        booking_type=created_booking["booking_type"],
        contact_preference=created_booking.get("contact_preference"),
        payment_required=created_booking["payment_required"],
        payment_amount=created_booking.get("payment_amount"),
        payment_url=created_booking.get("payment_url"),
        payment_status=created_booking.get("payment_status", PaymentStatus.PENDING),
        payment_reference=created_booking.get("payment_reference"),
        flutterwave_tx_ref=created_booking.get("flutterwave_tx_ref"),
        createdAt=created_booking["createdAt"],
        updatedAt=created_booking["updatedAt"],
        service=service_obj,
        tier=tier_obj
    )

@app.get("/bookings", response_model=List[Booking])
async def get_bookings(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        booking_type: Optional[BookingType] = None,
        current_user: UserInDB = Depends(get_current_active_user)
):
    query = {}
    if current_user.role != "admin":
        query["userId"] = str(current_user.id)
    if status:
        query["status"] = status
    if booking_type:
        query["booking_type"] = booking_type

    bookings = list(db.bookings.find(query).skip(skip).limit(limit))
    result = []

    for booking in bookings:
        try:
            service_obj = None
            tier_obj = None

            if booking.get("serviceId"):
                service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                if service:
                    service_obj = Service(
                        id=str(service["_id"]),
                        name=service["name"],
                        description=service["description"],
                        image=service.get("image"),
                        duration=service["duration"],
                        isAvailable=service["isAvailable"],
                        features=service.get("features", []),
                        requirements=service.get("requirements", []),
                        category_id=service.get("category_id"),
                        tier_id=service.get("tier_id"),
                        createdAt=service["createdAt"],
                        updatedAt=service["updatedAt"]
                    )

            if booking.get("tierId"):
                tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
                if tier:
                    # Get services for this tier
                    tier_services = list(db.services.find({"tier_id": booking["tierId"]}))
                    tier_service_objects = []

                    for service in tier_services:
                        tier_service_objects.append(
                            Service(
                                id=str(service["_id"]),
                                name=service["name"],
                                description=service["description"],
                                image=service.get("image"),
                                duration=service["duration"],
                                isAvailable=service["isAvailable"],
                                features=service.get("features", []),
                                requirements=service.get("requirements", []),
                                category_id=service.get("category_id"),
                                tier_id=service.get("tier_id"),
                                createdAt=service["createdAt"],
                                updatedAt=service["updatedAt"]
                            )
                        )

                    tier_obj = ServiceTier(
                        id=str(tier["_id"]),
                        name=tier["name"],
                        description=tier["description"],
                        price=tier["price"],
                        category_id=tier["category_id"],
                        image=tier.get("image"),
                        features=tier.get("features", []),
                        is_popular=tier.get("is_popular", False),
                        is_available=tier.get("is_available", True),
                        created_at=tier["created_at"],
                        updated_at=tier["updated_at"],
                        services=tier_service_objects
                    )

            result.append(
                Booking(
                    id=str(booking["_id"]),
                    userId=booking["userId"],
                    serviceId=booking.get("serviceId"),
                    tierId=booking.get("tierId"),
                    bookingDate=booking["bookingDate"],
                    status=booking["status"],
                    specialRequests=booking.get("specialRequests"),
                    booking_type=booking.get("booking_type", BookingType.CONSULTATION),
                    contact_preference=booking.get("contact_preference"),
                    payment_required=booking.get("payment_required", False),
                    payment_amount=booking.get("payment_amount"),
                    payment_url=booking.get("payment_url"),
                    payment_status=booking.get("payment_status", PaymentStatus.PENDING),
                    payment_reference=booking.get("payment_reference"),
                    flutterwave_tx_ref=booking.get("flutterwave_tx_ref"),
                    createdAt=booking["createdAt"],
                    updatedAt=booking["updatedAt"],
                    service=service_obj,
                    tier=tier_obj
                )
            )
        except Exception as e:
            logger.error(f"Error processing booking: {e}")

    return result


# Enhanced Airtable booking with tier support and payment integration
@app.post("/bookings/airtable", response_model=Booking)
async def create_airtable_booking(
        booking_form: AirtableBookingForm,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        if not booking_form.clientName.strip() or booking_form.clientName.lower() == "string":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client name must be a valid name, not 'string' or empty"
            )

        # Must have either serviceId or tierId, but not both
        if not booking_form.serviceId and not booking_form.tierId:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either serviceId or tierId must be provided"
            )

        if booking_form.serviceId and booking_form.tierId:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot book both service and tier simultaneously"
            )

        booking_email = current_user.email

        user = db.users.find_one({"email": booking_email})
        if not user:
            user_dict = {
                "email": booking_email,
                "firstName": booking_form.clientName.split(" ")[0],
                "lastName": " ".join(booking_form.clientName.split(" ")[1:]) if len(
                    booking_form.clientName.split(" ")) > 1 else "",
                "phone": booking_form.phone,
                "role": "user",
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
                "hashed_password": get_password_hash(str(uuid.uuid4()))
            }
            user_result = db.users.insert_one(user_dict)
            user_id = str(user_result.inserted_id)
        else:
            user_id = str(user["_id"])

        # Determine booking type and details
        service_name = ""
        category_name = ""
        booking_type = BookingType.CONSULTATION
        payment_required = False
        payment_amount = 0.0

        if booking_form.serviceId:
            # Individual service booking
            service = db.services.find_one({"_id": ObjectId(booking_form.serviceId)})
            if not service:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Service not found"
                )

            if not service["isAvailable"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Service is not available"
                )

            category = db.service_categories.find_one({"_id": ObjectId(service["category_id"])})
            if not category:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Service category not found"
                )

            service_name = service["name"]
            category_name = category["name"]
            booking_type = BookingType.CONSULTATION
            payment_required = False
            payment_amount = 0.0

        elif booking_form.tierId:
            # Tier booking
            tier = db.service_tiers.find_one({"_id": ObjectId(booking_form.tierId)})
            if not tier:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Service tier not found"
                )

            if not tier["is_available"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Service tier is not available"
                )

            category = db.service_categories.find_one({"_id": ObjectId(tier["category_id"])})
            if not category:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Service category not found"
                )

            service_name = tier["name"]
            category_name = category["name"]
            booking_type = BookingType.TIER_BOOKING
            payment_required = True
            payment_amount = tier["price"]

        booking_data = BookingInDB(
            userId=user_id,
            serviceId=booking_form.serviceId,
            tierId=booking_form.tierId,
            bookingDate=booking_form.bookingDate,
            status="pending",
            specialRequests=booking_form.specialRequests,
            booking_type=booking_type,
            payment_required=payment_required,
            payment_amount=payment_amount,
            payment_status=PaymentStatus.PENDING
        )
        result = db.bookings.insert_one(booking_data.dict(by_alias=True))
        created_booking = db.bookings.find_one({"_id": result.inserted_id})

        # Generate payment URL for tier bookings
        payment_url = None
        if booking_type == BookingType.TIER_BOOKING and payment_required:
            try:
                user_obj = get_user_by_id(user_id)
                if user_obj:
                    payment_url = generate_flutterwave_payment_url(
                        {"id": str(created_booking["_id"])},
                        user_obj.__dict__,
                        payment_amount
                    )

                    # Update booking with payment URL
                    db.bookings.update_one(
                        {"_id": result.inserted_id},
                        {"$set": {"payment_url": payment_url}}
                    )

                    # Update the created_booking dict for response
                    created_booking["payment_url"] = payment_url

            except Exception as e:
                logger.error(f"Payment URL generation failed for Airtable booking: {e}")
                # Continue without payment URL

        airtable_data = {
            "Booking ID": str(created_booking["_id"]),
            "Client Name": booking_form.clientName,
            "Service Requested": f"{service_name} ({category_name})",
            "Booking Date": booking_form.bookingDate.strftime("%Y-%m-%d"),
            "Booking Details": booking_form.specialRequests or "",
            "Status": "Pending",
            "Total Cost": float(payment_amount),
            "Booking Type": booking_type,
            "Payment Required": payment_required,
            "Payment URL": payment_url or "",
            "Feedback": [],
            "Subscription Plan": [],
            "User": []
        }
        logger.info(f"Airtable data prepared: {airtable_data}")

        airtable_response = add_to_airtable(airtable_data)
        logger.info(f"Airtable response: {airtable_response}")

        crm_client = CRMClientInDB(
            clientName=booking_form.clientName,
            contactInfo={"email": booking_email, "phone": booking_form.phone or ""},
            serviceBooked=booking_form.serviceId or booking_form.tierId,
            status="pending"
        )
        db.crm_clients.insert_one(crm_client.dict(by_alias=True))

        user_obj = get_user_by_id(user_id)
        if not user_obj:
            logger.error(f"User not found after creation: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User creation failed"
            )

        # Send appropriate emails based on booking type
        if booking_type == BookingType.CONSULTATION:
            confirmation_html = f"""
            <html>
                <body>
                    <h1>Service Request Received</h1>
                    <p>Dear {booking_form.clientName},</p>
                    <p>We have received your request for {service_name} from {category_name}.</p>
                    <p>Our team will contact you within 24 hours to discuss your requirements and provide a custom quote.</p>
                    <p>Booking Details:</p>
                    <ul>
                        <li>Service: {service_name}</li>
                        <li>Category: {category_name}</li>
                        <li>Date: {booking_form.bookingDate.strftime("%Y-%m-%d %H:%M")}</li>
                        <li>Status: Pending</li>
                    </ul>
                    <p>Best regards,<br>The Naija Concierge Team</p>
                </body>
            </html>
            """
            notification_message = f"""
            New service request:
            - Client: {booking_form.clientName}
            - Service: {service_name}
            - Category: {category_name}
            - Date: {booking_form.bookingDate.strftime("%Y-%m-%d %H:%M")}
            - Email: {booking_email}
            - Type: Contact Required
            """
        else:  # TIER_BOOKING
            payment_info = f"<p>Payment URL: <a href='{payment_url}'>Complete Payment</a></p>" if payment_url else "<p>Payment URL will be provided shortly.</p>"
            confirmation_html = f"""
            <html>
                <body>
                    <h1>Tier Booking Confirmation - Payment Required</h1>
                    <p>Dear {booking_form.clientName},</p>
                    <p>Your booking for {service_name} from {category_name} has been received.</p>
                    <p>To confirm your booking, please complete the payment of ₦{payment_amount}.</p>
                    {payment_info}
                    <p>Booking Details:</p>
                    <ul>
                        <li>Tier: {service_name}</li>
                        <li>Category: {category_name}</li>
                        <li>Date: {booking_form.bookingDate.strftime("%Y-%m-%d %H:%M")}</li>
                        <li>Amount: ₦{payment_amount}</li>
                        <li>Status: Pending Payment</li>
                    </ul>
                    <p>Best regards,<br>The Naija Concierge Team</p>
                </body>
            </html>
            """
            notification_message = f"""
            New tier booking:
            - Client: {booking_form.clientName}
            - Tier: {service_name}
            - Category: {category_name}
            - Date: {booking_form.bookingDate.strftime("%Y-%m-%d %H:%M")}
            - Email: {booking_email}
            - Amount: ₦{payment_amount}
            - Payment Required: Yes
            - Payment URL: {payment_url or "Generation failed"}
            """

        send_email(booking_email, "Booking Confirmation - Naija Concierge", confirmation_html)
        send_admin_notification("New Booking Received", notification_message)

        # Get service and tier objects for response
        service_obj = None
        tier_obj = None

        if booking_form.serviceId:
            service = db.services.find_one({"_id": ObjectId(booking_form.serviceId)})
            if service:
                service_obj = Service(
                    id=str(service["_id"]),
                    name=service["name"],
                    description=service["description"],
                    image=service.get("image"),
                    duration=service["duration"],
                    isAvailable=service["isAvailable"],
                    features=service.get("features", []),
                    requirements=service.get("requirements", []),
                    category_id=service.get("category_id"),
                    tier_id=service.get("tier_id"),
                    createdAt=service["createdAt"],
                    updatedAt=service["updatedAt"]
                )

        if booking_form.tierId:
            tier = db.service_tiers.find_one({"_id": ObjectId(booking_form.tierId)})
            if tier:
                tier_obj = ServiceTier(
                    id=str(tier["_id"]),
                    name=tier["name"],
                    description=tier["description"],
                    price=tier["price"],
                    category_id=tier["category_id"],
                    image=tier.get("image"),
                    features=tier.get("features", []),
                    is_popular=tier.get("is_popular", False),
                    is_available=tier.get("is_available", True),
                    created_at=tier["created_at"],
                    updated_at=tier["updated_at"],
                    services=[]
                )

        return Booking(
            id=str(created_booking["_id"]),
            userId=created_booking["userId"],
            serviceId=created_booking.get("serviceId"),
            tierId=created_booking.get("tierId"),
            bookingDate=created_booking["bookingDate"],
            status=created_booking["status"],
            specialRequests=created_booking.get("specialRequests"),
            booking_type=created_booking["booking_type"],
            contact_preference=created_booking.get("contact_preference"),
            payment_required=created_booking["payment_required"],
            payment_amount=created_booking.get("payment_amount"),
            payment_url=created_booking.get("payment_url"),
            payment_status=created_booking.get("payment_status", PaymentStatus.PENDING),
            payment_reference=created_booking.get("payment_reference"),
            flutterwave_tx_ref=created_booking.get("flutterwave_tx_ref"),
            createdAt=created_booking["createdAt"],
            updatedAt=created_booking["updatedAt"],
            service=service_obj,
            tier=tier_obj
        )
    except Exception as e:
        logger.error(f"Error creating Airtable booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create booking: {str(e)}"
        )


# Payment webhook endpoint
@app.post("/webhooks/flutterwave")
async def flutterwave_webhook(request: Request):
    """Handle Flutterwave payment webhooks"""
    try:
        # Get raw body for signature verification
        body = await request.body()
        signature = request.headers.get("verif-hash")

        if not verify_webhook_signature(body.decode(), signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid signature"
            )

        # Parse webhook data
        webhook_data = json.loads(body.decode())
        event = webhook_data.get("event")
        data = webhook_data.get("data", {})

        if event == "charge.completed":
            tx_ref = data.get("tx_ref")
            flw_ref = data.get("flw_ref")
            payment_status = data.get("status")
            amount = data.get("amount")

            if not tx_ref:
                logger.error("No transaction reference in webhook")
                return {"status": "error", "message": "No transaction reference"}

            # Find booking by transaction reference
            booking = db.bookings.find_one({"flutterwave_tx_ref": tx_ref})
            if not booking:
                logger.error(f"No booking found for tx_ref: {tx_ref}")
                return {"status": "error", "message": "Booking not found"}

            # Verify payment with Flutterwave
            try:
                verification_response = verify_flutterwave_payment(tx_ref)
                verified_data = verification_response.get("data", {})
                verified_status = verified_data.get("status")
                verified_amount = verified_data.get("amount")

                if verified_status == "successful" and verified_amount == booking.get("payment_amount"):
                    # Update booking status
                    update_data = {
                        "payment_status": PaymentStatus.SUCCESSFUL,
                        "payment_reference": flw_ref,
                        "status": "confirmed",
                        "updatedAt": datetime.utcnow()
                    }

                    db.bookings.update_one(
                        {"_id": ObjectId(booking["_id"])},
                        {"$set": update_data}
                    )

                    # Send confirmation email
                    user = get_user_by_id(booking["userId"])
                    if user:
                        # Get tier/service details
                        tier_name = "Unknown"
                        if booking.get("tierId"):
                            tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
                            if tier:
                                tier_name = tier["name"]

                        confirmation_html = f"""
                        <html>
                            <body>
                                <h1>Payment Successful - Booking Confirmed</h1>
                                <p>Dear {user.firstName},</p>
                                <p>Your payment has been successfully processed and your booking is now confirmed.</p>
                                <p>Payment Details:</p>
                                <ul>
                                    <li>Tier: {tier_name}</li>
                                    <li>Amount: ₦{verified_amount}</li>
                                    <li>Reference: {flw_ref}</li>
                                    <li>Status: Confirmed</li>
                                </ul>
                                <p>Our team will contact you shortly to coordinate the service delivery.</p>
                                <p>Best regards,<br>The Naija Concierge Team</p>
                            </body>
                        </html>
                        """
                        send_email(user.email, "Payment Successful - Booking Confirmed", confirmation_html)

                    # Send admin notification
                    admin_notification = f"""
                    Payment received for booking:
                    - Booking ID: {booking["_id"]}
                    - Client: {user.firstName} {user.lastName} if user else "Unknown"
                    - Tier: {tier_name}
                    - Amount: ₦{verified_amount}
                    - Reference: {flw_ref}
                    - Status: Confirmed
                    """
                    send_admin_notification("Payment Received", admin_notification)

                    logger.info(f"Payment successful for booking {booking['_id']}")

                elif verified_status == "failed":
                    # Update booking as failed
                    db.bookings.update_one(
                        {"_id": ObjectId(booking["_id"])},
                        {"$set": {
                            "payment_status": PaymentStatus.FAILED,
                            "payment_reference": flw_ref,
                            "updatedAt": datetime.utcnow()
                        }}
                    )

                    logger.info(f"Payment failed for booking {booking['_id']}")

                else:
                    logger.warning(f"Unhandled payment status: {verified_status}")

            except Exception as e:
                logger.error(f"Error verifying payment: {e}")
                return {"status": "error", "message": "Payment verification failed"}

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed"
        )


# Payment verification endpoint
@app.get("/payments/verify/{tx_ref}")
async def verify_payment(
        tx_ref: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Manually verify payment status"""
    try:
        # Find booking by transaction reference
        booking = db.bookings.find_one({"flutterwave_tx_ref": tx_ref})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        # Check if user has permission to verify this payment
        if current_user.role != "admin" and booking["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )

        # Verify with Flutterwave
        verification_response = verify_flutterwave_payment(tx_ref)
        verified_data = verification_response.get("data", {})
        verified_status = verified_data.get("status")
        verified_amount = verified_data.get("amount")

        return {
            "tx_ref": tx_ref,
            "status": verified_status,
            "amount": verified_amount,
            "booking_id": str(booking["_id"]),
            "verification_data": verified_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying payment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Payment verification failed"
        )


# Payment success/failure redirect pages
@app.get("/booking/payment-success")
async def payment_success(tx_ref: Optional[str] = None):
    """Payment success redirect page"""
    return {
        "message": "Payment successful! Your booking has been confirmed.",
        "tx_ref": tx_ref,
        "status": "success"
    }


@app.get("/booking/payment-failed")
async def payment_failed(tx_ref: Optional[str] = None):
    """Payment failure redirect page"""
    return {
        "message": "Payment failed. Please try again or contact support.",
        "tx_ref": tx_ref,
        "status": "failed"
    }


# Contact routes
@app.post("/contact")
async def send_contact_message(message: ContactMessage):
    try:
        if not message.name.strip() or not message.subject.strip() or not message.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name, subject, and message are required"
            )

        message_dict = message.dict()
        message_dict["createdAt"] = datetime.utcnow()
        db.contact_messages.insert_one(message_dict)

        contact_html = f"""
        <html>
            <body>
                <h1>New Contact Message</h1>
                <p><strong>Name:</strong> {message.name}</p>
                <p><strong>Email:</strong> {message.email}</p>
                <p><strong>Phone:</strong> {message.phone or "Not provided"}</p>
                <p><strong>Subject:</strong> {message.subject}</p>
                <p><strong>Message:</strong></p>
                <p>{message.message}</p>
            </body>
        </html>
        """
        send_admin_notification(f"New Contact Message: {message.subject}", contact_html)

        confirmation_html = f"""
        <html>
            <body>
                <h1>Thank You for Contacting Us</h1>
                <p>Dear {message.name},</p>
                <p>We have received your message and will get back to you shortly.</p>
                <p><strong>Subject:</strong> {message.subject}</p>
                <p><strong>Message:</strong></p>
                <p>{message.message}</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """
        send_email(message.email, "Thank You for Contacting Naija Concierge", confirmation_html)

        return {"message": "Contact message sent successfully"}
    except Exception as e:
        logger.error(f"Error sending contact message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send contact message"
        )


# Newsletter subscription
@app.post("/newsletter/subscribe")
async def subscribe_to_newsletter(email: EmailStr):
    try:
        existing_subscriber = db.newsletter_subscribers.find_one({"email": email})
        if existing_subscriber:
            if existing_subscriber["is_active"]:
                return {"message": "Email is already subscribed to the newsletter"}
            else:
                db.newsletter_subscribers.update_one(
                    {"email": email},
                    {"$set": {"is_active": True, "subscribed_at": datetime.utcnow()}}
                )
                return {"message": "Newsletter subscription reactivated successfully"}

        subscriber = NewsletterSubscriberInDB(email=email)
        db.newsletter_subscribers.insert_one(subscriber.dict(by_alias=True))

        welcome_html = f"""
        <html>
            <body>
                <h1>Welcome to Naija Concierge Newsletter!</h1>
                <p>Thank you for subscribing to our newsletter.</p>
                <p>You'll receive updates about our latest services, special offers, and Lagos lifestyle tips.</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """
        send_email(email, "Welcome to Naija Concierge Newsletter", welcome_html)

        return {"message": "Successfully subscribed to newsletter"}
    except Exception as e:
        logger.error(f"Error subscribing to newsletter: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to subscribe to newsletter"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
