from fastapi import FastAPI, Depends, HTTPException, status, Request, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, GetJsonSchemaHandler, BeforeValidator, ValidationError, field_validator
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
import httpx
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

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
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://sorted-concierge.vercel.app/")

EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
EXCHANGE_RATE_BASE_URL = "https://v6.exchangerate-api.com/v6"

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


oauth = OAuth()

# Configure Google OAuth
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

app = FastAPI(title="Naija Concierge API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="session"
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
    hashed_password: Optional[str] = None

    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str
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



class GoogleUserCreate(BaseModel):
    google_token: str

class GoogleUserInfo(BaseModel):
    email: EmailStr
    firstName: str
    lastName: str
    profileImage: Optional[str] = None


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


class BlogBase(BaseModel):
    title: str
    slug: str
    content: str
    excerpt: str
    coverImage: Optional[str] = None
    author: Dict[str, str]
    tags: List[str]


class BlogCreate(BlogBase):
    pass


class BlogUpdate(BaseModel):
    title: Optional[str] = None
    slug: Optional[str] = None
    content: Optional[str] = None
    excerpt: Optional[str] = None
    coverImage: Optional[str] = None
    author: Optional[Dict[str, str]] = None
    tags: Optional[List[str]] = None


class BlogInDB(BlogBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Blog(BlogBase):
    id: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True


class EmergencyAlertBase(BaseModel):
    userId: str
    message: str
    location: Optional[str] = None
    status: str = "pending"


class EmergencyAlertCreate(EmergencyAlertBase):
    pass


class EmergencyAlertUpdate(BaseModel):
    message: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None


class EmergencyAlertInDB(EmergencyAlertBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class EmergencyAlert(EmergencyAlertBase):
    id: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True



class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    subject: str
    message: str



class BookingDataPoint(BaseModel):
    name: str
    bookings: int
    completed: int

class RevenueDataPoint(BaseModel):
    name: str
    revenue: float

class ChartDataResponse(BaseModel):
    bookingData: List[BookingDataPoint]
    revenueData: List[RevenueDataPoint]


class Timeframe(str, Enum):
    weekly = "weekly"
    monthly = "monthly"


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



class GalleryImageBase(BaseModel):
    title: str
    description: Optional[str] = None
    category: str
    tags: List[str] = []

class GalleryImageCreate(GalleryImageBase):
    pass

class GalleryImageUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class GalleryImageInDB(GalleryImageBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    image_url: str
    created_by: str  # User ID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class GalleryImage(GalleryImageBase):
    id: str
    image_url: str
    created_by: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class BookingReschedule(BaseModel):
    new_booking_date: datetime
    reason: Optional[str] = None

class BookingCancellation(BaseModel):
    reason: Optional[str] = None

class UserBookingStats(BaseModel):
    total_bookings: int
    pending_bookings: int
    confirmed_bookings: int
    completed_bookings: int
    cancelled_bookings: int
    total_spent: float
    upcoming_bookings: int

class BookingHistory(BaseModel):
    booking: Booking
    status_history: List[Dict[str, Any]] = []



class BookingStatusUpdate(BaseModel):
    status: str
    admin_notes: Optional[str] = None
    notify_user: bool = True


class BookingAssignment(BaseModel):
    assigned_to: Optional[str] = None  # Staff/vendor ID or name
    assignment_notes: Optional[str] = None
    notify_assignee: bool = True


class AdminBookingUpdate(BaseModel):
    bookingDate: Optional[datetime] = None
    status: Optional[str] = None
    specialRequests: Optional[str] = None
    contact_preference: Optional[str] = None
    payment_status: Optional[PaymentStatus] = None
    payment_amount: Optional[float] = None
    admin_notes: Optional[str] = None
    assigned_to: Optional[str] = None
    priority: Optional[str] = None  # low, medium, high, urgent


class BookingAdminNote(BaseModel):
    note: str
    note_type: Optional[str] = "general"  # general, important, warning, follow_up


class BulkBookingAction(BaseModel):
    booking_ids: List[str]
    action: str  # update_status, assign, delete, etc.
    data: Dict[str, Any]


class BookingRevenueStats(BaseModel):
    total_revenue: float
    monthly_revenue: float
    daily_average: float
    top_services: List[Dict[str, Any]]
    payment_breakdown: Dict[str, Any]


class BookingAdvancedFilters(BaseModel):
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    status: Optional[str] = None
    booking_type: Optional[BookingType] = None
    payment_status: Optional[PaymentStatus] = None
    assigned_to: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    has_special_requests: Optional[bool] = None
    priority: Optional[str] = None



class AdminStats(BaseModel):
    totalUsers: int
    totalBookings: int
    totalBookingRevenue: float
    totalServiceTiers: int
    userGrowth: float
    bookingGrowth: float
    revenueGrowth: float
    tierGrowth: float





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

    if not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please sign in with Google"
        )

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


async def validate_google_token(token: str) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v3/tokeninfo",
                params={"id_token": token}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Google token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token"
        )

def create_user_from_google(google_user: dict) -> dict:
    """Create user dictionary from Google user info"""
    return {
        "email": google_user["email"],
        "firstName": google_user.get("given_name", ""),
        "lastName": google_user.get("family_name", ""),
        "profileImage": google_user.get("picture"),
        "role": "user",
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow(),
        "hashed_password": None  # No password for Google users
    }


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
# def generate_flutterwave_payment_url(booking_data: Dict, user_data: Dict, amount: float) -> str:
#     """Generate Flutterwave payment URL for tier bookings"""
#     if not FLUTTERWAVE_SECRET_KEY:
#         logger.error("Flutterwave secret key not configured")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Payment service not configured"
#         )
#
#     try:
#         # Generate unique transaction reference
#         tx_ref = f"booking_{booking_data['id']}_{uuid.uuid4().hex[:8]}"
#
#         # Flutterwave payment payload
#         payment_payload = {
#             "tx_ref": tx_ref,
#             "amount": amount,
#             "currency": "NGN",
#             "redirect_url": f"{FRONTEND_URL}/booking/confirmation",
#             "payment_options": "card,banktransfer,ussd",
#             "customer": {
#                 "email": user_data["email"],
#                 "phonenumber": user_data.get("phone", ""),
#                 "name": f"{user_data['firstName']} {user_data['lastName']}"
#             },
#             "customizations": {
#                 "title": "Naija Concierge - Tier Booking",
#                 "description": f"Payment for tier booking",
#                 "logo": "https://your-logo-url.com/logo.png"
#             },
#             "meta": {
#                 "booking_id": str(booking_data["id"]),
#                 "user_id": user_data["id"],
#                 "booking_type": "tier_booking"
#             }
#         }
#
#         headers = {
#             "Authorization": f"Bearer {FLUTTERWAVE_SECRET_KEY}",
#             "Content-Type": "application/json"
#         }
#
#         response = requests.post(
#             f"{FLUTTERWAVE_BASE_URL}/payments",
#             json=payment_payload,
#             headers=headers
#         )
#         response.raise_for_status()
#
#         payment_data = response.json()
#         if payment_data["status"] == "success":
#             # Update booking with transaction reference
#             db.bookings.update_one(
#                 {"_id": ObjectId(booking_data["id"])},
#                 {"$set": {"flutterwave_tx_ref": tx_ref}}
#             )
#             return payment_data["data"]["link"]
#         else:
#             logger.error(f"Payment URL generation failed: {payment_data}")
#             raise Exception(f"Payment URL generation failed: {payment_data}")
#
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Flutterwave API request error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to generate payment URL"
#         )
#     except Exception as e:
#         logger.error(f"Flutterwave payment URL generation error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to generate payment URL"
#         )
#

def generate_flutterwave_payment_url(
    booking_data: Dict,
    user_data: Dict,
    amount: float,
    currency: str = "NGN"
) -> str:
    """Generate Flutterwave payment URL with proper currency support"""
    if not FLUTTERWAVE_SECRET_KEY:
        logger.error("Flutterwave secret key not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Payment service not configured"
        )

    try:
        # Generate unique transaction reference
        tx_ref = f"booking_{booking_data['id']}_{uuid.uuid4().hex[:8]}"

        # Ensure amount is properly formatted for the currency
        if currency == "NGN":
            # NGN amounts should be whole numbers
            formatted_amount = int(amount)
        else:
            # Other currencies can have decimals
            formatted_amount = round(amount, 2)

        # Flutterwave payment payload with proper currency handling
        payment_payload = {
            "tx_ref": tx_ref,
            "amount": formatted_amount,
            "currency": currency,
            "redirect_url": f"{FRONTEND_URL}/booking/payment-success?tx_ref={tx_ref}",
            "payment_options": "card,banktransfer,ussd" if currency == "NGN" else "card",
            "customer": {
                "email": user_data["email"],
                "phonenumber": user_data.get("phone", ""),
                "name": f"{user_data['firstName']} {user_data['lastName']}"
            },
            "customizations": {
                "title": "Naija Concierge - Tier Booking",
                "description": f"Payment for tier booking ({currency} {formatted_amount})",
                "logo": "https://your-logo-url.com/logo.png"
            },
            "meta": {
                "booking_id": str(booking_data["id"]),
                "user_id": user_data["id"],
                "booking_type": "tier_booking",
                "original_currency": "NGN",
                "payment_currency": currency,
                "original_amount": booking_data.get("original_amount", amount)
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
            # Update booking with transaction reference and currency info
            db.bookings.update_one(
                {"_id": ObjectId(booking_data["id"])},
                {"$set": {
                    "flutterwave_tx_ref": tx_ref,
                    "payment_currency": currency,
                    "payment_amount_original": booking_data.get("original_amount", amount),
                    "payment_amount_converted": formatted_amount
                }}
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


async def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Get real-time exchange rate between currencies"""
    if from_currency == to_currency:
        return 1.0

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{EXCHANGE_RATE_BASE_URL}/{EXCHANGE_RATE_API_KEY}/latest/{from_currency}"
            )
            response.raise_for_status()
            rates = response.json().get("conversion_rates", {})
            return rates.get(to_currency, 1.0)
        except Exception as e:
            logger.error(f"Error fetching exchange rate: {e}")
            # Fallback rates if API fails
            fallback_rates = {
                "NGN": {"USD": 0.00065, "EUR": 0.0006, "GBP": 0.00052},
                "USD": {"NGN": 1538.46, "EUR": 0.92, "GBP": 0.80},
                "EUR": {"NGN": 1666.67, "USD": 1.09, "GBP": 0.87},
                "GBP": {"NGN": 1923.08, "USD": 1.25, "EUR": 1.15}
            }
            return fallback_rates.get(from_currency, {}).get(to_currency, 1.0)


async def convert_price(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert price from one currency to another"""
    if from_currency == to_currency:
        return amount

    exchange_rate = await get_exchange_rate(from_currency, to_currency)
    converted_amount = amount * exchange_rate
    return round(converted_amount, 2)




# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Naija Concierge API"}



@app.post("/auth/register", response_model=Token)
async def register(user: UserCreate):
    # Check if email is already registered
    if db.users.find_one({"email": user.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Hash the password
    hashed_password = get_password_hash(user.password)

    # Prepare user data for database insertion
    user_dict = user.dict()
    del user_dict["password"]
    user_dict["hashed_password"] = hashed_password
    user_dict["role"] = "user"
    user_dict["createdAt"] = datetime.utcnow()
    user_dict["updatedAt"] = datetime.utcnow()

    # Insert user into database directly
    try:
        result = db.users.insert_one(user_dict)
    except Exception as e:
        logger.error(f"Database insertion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

    # Fetch the created user
    created_user = db.users.find_one({"_id": result.inserted_id})
    if not created_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve created user"
        )

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

    # Generate access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    # Send welcome email
    welcome_html = f"""
    <html>
        <body>
            <h1>Welcome to Sorted Concierge, {user.firstName}!</h1>
            <p>Thank you for registering with us. We're excited to help you experience Luxury like never before.</p>
            <p>If you have any questions or need assistance, please don't hesitate to contact us.</p>
            <p>Best regards,<br>The Naija Concierge Team</p>
        </body>
    </html>
    """
    send_email(user.email, "Welcome to Sorted Concierge", welcome_html)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }


@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Handle Google-authenticated users trying to use password login
    if not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account was created with Google. Please sign in with Google.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(form_data.password, user.hashed_password):
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



@app.get("/auth/google/login")
async def login_google(request: Request):
    # Absolute redirect URL for Google OAuth
    redirect_uri = request.url_for('auth_google_callback')
    return await oauth.google.authorize_redirect(request, str(redirect_uri))


# @app.get("/auth/google/callback")
# async def auth_google_callback(request: Request):
#     try:
#         token = await oauth.google.authorize_access_token(request)
#     except Exception as e:
#         logger.error(f"Google OAuth error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Failed to authenticate with Google"
#         )
#
#     # Get user info from Google
#     user_info = token.get('userinfo')
#     if not user_info:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Failed to get user info from Google"
#         )
#
#     email = user_info.get('email')
#     if not email:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Email not provided by Google"
#         )
#
#     # Check if user exists
#     user = db.users.find_one({"email": email})
#
#     if not user:
#         # Create new user from Google info
#         user_data = {
#             "email": email,
#             "firstName": user_info.get('given_name', ''),
#             "lastName": user_info.get('family_name', ''),
#             "profileImage": user_info.get('picture'),
#             "role": "user",
#             "createdAt": datetime.utcnow(),
#             "updatedAt": datetime.utcnow(),
#             "hashed_password": ""  # No password for Google users
#         }
#
#         try:
#             result = db.users.insert_one(user_data)
#             user_data["_id"] = result.inserted_id
#             user = user_data
#         except Exception as e:
#             logger.error(f"Failed to create user from Google auth: {e}")
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Failed to create user account"
#             )
#
#     # Convert to UserInDB model
#     user_in_db = UserInDB(
#         id=user["_id"],
#         email=user["email"],
#         firstName=user.get("firstName", ""),
#         lastName=user.get("lastName", ""),
#         phone=user.get("phone"),
#         address=user.get("address"),
#         profileImage=user.get("profileImage"),
#         role=user["role"],
#         createdAt=user["createdAt"],
#         updatedAt=user["updatedAt"],
#         hashed_password=user.get("hashed_password", "")
#     )
#
#     # Create JWT token (same as regular login)
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": email}, expires_delta=access_token_expires
#     )
#
#     # Prepare user response
#     user_response = User(
#         id=str(user_in_db.id),
#         email=user_in_db.email,
#         firstName=user_in_db.firstName,
#         lastName=user_in_db.lastName,
#         phone=user_in_db.phone,
#         address=user_in_db.address,
#         profileImage=user_in_db.profileImage,
#         role=user_in_db.role,
#         createdAt=user_in_db.createdAt,
#         updatedAt=user_in_db.updatedAt
#     )
#
#     return {
#         "access_token": access_token,
#         "token_type": "bearer",
#         "user": user_response
#     }




# @app.get("/auth/google/callback")
# async def auth_google_callback(request: Request):
#     try:
#         token = await oauth.google.authorize_access_token(request)
#     except Exception as e:
#         logger.error(f"Google OAuth error: {e}")
#         # Redirect to frontend with error
#         error_params = urlencode({
#             "error": "oauth_failed",
#             "message": "Failed to authenticate with Google"
#         })
#         return RedirectResponse(
#             f"{FRONTEND_URL}/auth/callback?{error_params}"
#         )
#
#     # Get user info from Google
#     user_info = token.get('userinfo')
#     if not user_info:
#         error_params = urlencode({
#             "error": "no_user_info",
#             "message": "Failed to get user info from Google"
#         })
#         return RedirectResponse(
#             f"{FRONTEND_URL}/auth/callback?{error_params}"
#         )
#
#     email = user_info.get('email')
#     if not email:
#         error_params = urlencode({
#             "error": "no_email",
#             "message": "Email not provided by Google"
#         })
#         return RedirectResponse(
#             f"{FRONTEND_URL}/auth/callback?{error_params}"
#         )
#
#     try:
#         # Check if user exists
#         user = db.users.find_one({"email": email})
#
#         if not user:
#             # Create new user from Google info
#             user_data = {
#                 "email": email,
#                 "firstName": user_info.get('given_name', ''),
#                 "lastName": user_info.get('family_name', ''),
#                 "profileImage": user_info.get('picture'),
#                 "role": "user",
#                 "createdAt": datetime.utcnow(),
#                 "updatedAt": datetime.utcnow(),
#                 "hashed_password": ""  # No password for Google users
#             }
#
#             result = db.users.insert_one(user_data)
#             user_data["_id"] = result.inserted_id
#             user = user_data
#
#         # Create JWT token
#         access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#         access_token = create_access_token(
#             data={"sub": email}, expires_delta=access_token_expires
#         )
#
#         # Prepare success redirect URL with token and user data
#         success_params = urlencode({
#             "token": access_token,
#             "user": json.dumps({
#                 "id": str(user["_id"]),
#                 "email": user["email"],
#                 "firstName": user.get("firstName", ""),
#                 "lastName": user.get("lastName", ""),
#                 "phone": user.get("phone"),
#                 "address": user.get("address"),
#                 "profileImage": user.get("profileImage"),
#                 "role": user["role"],
#                 "createdAt": user["createdAt"].isoformat(),
#                 "updatedAt": user["updatedAt"].isoformat()
#             })
#         })
#
#         return RedirectResponse(
#             f"{FRONTEND_URL}/auth/callback?{success_params}"
#         )
#
#     except Exception as e:
#         logger.error(f"Error processing Google auth: {e}")
#         error_params = urlencode({
#             "error": "processing_error",
#             "message": "Failed to process authentication"
#         })
#         return RedirectResponse(
#             f"{FRONTEND_URL}/auth/callback?{error_params}"
#         )
#
# @app.get("/auth/google/callback/success")
# async def google_auth_success(request: Request, token: str, user: str):
#     user_data = json.loads(user)
#     return JSONResponse({
#         "type": "google-auth-success",
#         "token": token,
#         "user": user_data
#     })
#
# @app.get("/auth/google/callback/error")
# async def google_auth_error(request: Request, message: str):
#     return JSONResponse({
#         "type": "google-auth-error",
#         "message": message
#     })
#
#
# @app.post("/auth/register/google", response_model=Token)
# async def register_with_google(google_user: GoogleUserCreate):
#     """
#     Register a new user using Google authentication.
#
#     Requires a valid Google ID token obtained from the frontend Google Sign-In flow.
#     """
#     # Validate Google token
#     try:
#         user_info = await validate_google_token(google_user.google_token)
#     except Exception as e:
#         logger.error(f"Google token validation error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid Google token"
#         )
#
#     # Check required fields
#     if not user_info.get("email"):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Email not provided by Google"
#         )
#
#     email = user_info["email"]
#
#     # Check if user already exists
#     if db.users.find_one({"email": email}):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Email already registered"
#         )
#
#     # Create new user
#     user_data = create_user_from_google(user_info)
#
#     try:
#         result = db.users.insert_one(user_data)
#         user_data["_id"] = result.inserted_id
#     except Exception as e:
#         logger.error(f"Database insertion error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create user"
#         )
#
#     # Convert to UserInDB model
#     user_in_db = UserInDB(**user_data)
#
#     # Generate access token
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": email}, expires_delta=access_token_expires
#     )
#
#     # Prepare user response
#     user_response = User(
#         id=str(user_in_db.id),
#         email=user_in_db.email,
#         firstName=user_in_db.firstName,
#         lastName=user_in_db.lastName,
#         phone=user_in_db.phone,
#         address=user_in_db.address,
#         profileImage=user_in_db.profileImage,
#         role=user_in_db.role,
#         createdAt=user_in_db.createdAt,
#         updatedAt=user_in_db.updatedAt
#     )
#
#     # Send welcome email
#     welcome_html = f"""
#     <html>
#         <body>
#             <h1>Welcome to Sorted Concierge, {user_in_db.firstName}!</h1>
#             <p>Thank you for registering with us using your Google account.</p>
#             <p>We're excited to help you experience Luxury like never before.</p>
#             <p>Best regards,<br>The Sorted Concierge Team</p>
#         </body>
#     </html>
#     """
#     send_email(user_in_db.email, "Welcome to Sorted Concierge", welcome_html)
#
#     return {
#         "access_token": access_token,
#         "token_type": "bearer",
#         "user": user_response
#     }


@app.get("/auth/google/login")
async def login_via_google(request: Request, redirect_uri: str, register: bool = False):
    """
    Initiates Google OAuth flow with proper redirect handling
    """
    # Validate redirect_uri for security
    allowed_domains = ["http://localhost:3000", "https://sorted-concierge.vercel.app", "https://thesortedconcierge.com"]
    if not any(redirect_uri.startswith(domain) for domain in allowed_domains):
        raise HTTPException(status_code=400, detail="Invalid redirect URI")

    # Store register flag in session
    request.session["google_register"] = register

    # Initiate Google OAuth flow
    return await oauth.google.authorize_redirect(
        request,
        f"{redirect_uri}?register=true" if register else redirect_uri
    )


@router.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    try:
        # Get register flag from session
        register = request.session.get("google_register", False)
        redirect_uri = request.query_params.get('redirect_uri', FRONTEND_URL)

        # Complete OAuth flow
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')

        if not user_info or not user_info.get('email'):
            return RedirectResponse(f"{redirect_uri}?error=Failed to get user info from Google")

        email = user_info['email']

        # Check if user exists or create new
        user = db.users.find_one({"email": email})

        if not user:
            if not register:
                return RedirectResponse(f"{redirect_uri}?error=Account not found. Please register first.")

            user_data = {
                "email": email,
                "firstName": user_info.get('given_name', ''),
                "lastName": user_info.get('family_name', ''),
                "profileImage": user_info.get('picture'),
                "role": "user",
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
                "hashed_password": ""
            }
            result = db.users.insert_one(user_data)
            user_data["_id"] = result.inserted_id
            user = user_data

        # Generate JWT token
        access_token = create_access_token(data={"sub": email})

        # Prepare user data for redirect
        user_response = {
            "id": str(user["_id"]),
            "email": user["email"],
            "firstName": user.get("firstName", ""),
            "lastName": user.get("lastName", ""),
            "profileImage": user.get("profileImage", ""),
            "role": user["role"]
        }

        params = {
            "token": access_token,
            "user": json.dumps(user_response),
            "register": "true" if register else "false"
        }

        return RedirectResponse(f"{redirect_uri}?{urlencode(params)}")

    except Exception as e:
        redirect_uri = request.query_params.get('redirect_uri', FRONTEND_URL)
        return RedirectResponse(f"{redirect_uri}?error={str(e)}")


@app.post("/auth/register/google")
async def register_with_google(google_user: GoogleUserCreate):
    try:
        # Validate Google token
        user_info = await validate_google_token(google_user.google_token)
        if not user_info.get("email"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not provided by Google"
            )

        email = user_info["email"]

        # Check if user exists
        if db.users.find_one({"email": email}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Create new user
        user_data = {
            "email": email,
            "firstName": user_info.get('given_name', ''),
            "lastName": user_info.get('family_name', ''),
            "profileImage": user_info.get('picture'),
            "role": "user",
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
            "hashed_password": ""
        }

        result = db.users.insert_one(user_data)
        user_data["_id"] = result.inserted_id

        # Generate token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email}, expires_delta=access_token_expires
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": str(user_data["_id"]),
                "email": user_data["email"],
                "firstName": user_data["firstName"],
                "lastName": user_data["lastName"],
                "profileImage": user_data["profileImage"],
                "role": user_data["role"]
            }
        }

    except Exception as e:
        logger.error(f"Google registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register with Google"
        )


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


@app.get("/users", response_model=List[User])
async def get_users(
        skip: int = 0,
        limit: int = 100,
        current_user: UserInDB = Depends(get_admin_user)
):
    users = list(db.users.find().skip(skip).limit(limit))
    return [
        User(
            id=str(user["_id"]),
            email=user["email"],
            firstName=user["firstName"],
            lastName=user["lastName"],
            phone=user.get("phone"),
            address=user.get("address"),
            profileImage=user.get("profileImage"),
            role=user["role"],
            createdAt=user["createdAt"],
            updatedAt=user["updatedAt"]
        ) for user in users
    ]


@app.get("/users/{user_id}", response_model=User)
async def get_user_details(
        user_id: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    # Only admins can view other users' details
    if str(current_user.id) != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return User(
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


@app.put("/users/{user_id}", response_model=User)
async def update_user(
        user_id: str,
        user_update: UserUpdate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    # Only the user themselves or an admin can update a user
    if str(current_user.id) != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    # Check if user exists
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Update user
    update_data = user_update.dict(exclude_unset=True)
    if update_data:
        update_data["updatedAt"] = datetime.utcnow()
        result = db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User update failed",
            )

    # Get updated user
    updated_user = get_user_by_id(user_id)

    return User(
        id=str(updated_user.id),
        email=updated_user.email,
        firstName=updated_user.firstName,
        lastName=updated_user.lastName,
        phone=updated_user.phone,
        address=updated_user.address,
        profileImage=updated_user.profileImage,
        role=updated_user.role,
        createdAt=updated_user.createdAt,
        updatedAt=updated_user.updatedAt
    )


@app.get("/users/me/bookings", response_model=List[Booking])
async def get_my_bookings(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = Query(None, description="Filter by booking status"),
        booking_type: Optional[BookingType] = Query(None, description="Filter by booking type"),
        date_from: Optional[datetime] = Query(None, description="Filter bookings from this date"),
        date_to: Optional[datetime] = Query(None, description="Filter bookings to this date"),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Get current user's bookings with filtering options"""
    try:
        query = {"userId": str(current_user.id)}

        # Apply filters
        if status:
            query["status"] = status
        if booking_type:
            query["booking_type"] = booking_type
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["$gte"] = date_from
            if date_to:
                date_filter["$lte"] = date_to
            query["bookingDate"] = date_filter

        # Sort by booking date (newest first)
        bookings = list(
            db.bookings.find(query)
            .sort("bookingDate", -1)
            .skip(skip)
            .limit(limit)
        )

        result = []
        for booking in bookings:
            try:
                service_obj = None
                tier_obj = None

                # Get service details if serviceId exists
                if booking.get("serviceId"):
                    service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                    if service:
                        # Get category for service
                        category = None
                        if service.get("category_id"):
                            category_doc = db.service_categories.find_one({"_id": ObjectId(service["category_id"])})
                            if category_doc:
                                category = ServiceCategory(
                                    id=str(category_doc["_id"]),
                                    name=category_doc["name"],
                                    description=category_doc["description"],
                                    category_type=category_doc["category_type"],
                                    image=category_doc.get("image"),
                                    is_active=category_doc["is_active"],
                                    created_at=category_doc["created_at"],
                                    updated_at=category_doc["updated_at"],
                                    tiers=[],
                                    services=[]
                                )

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
                            updatedAt=service["updatedAt"],
                            category=category
                        )

                # Get tier details if tierId exists
                if booking.get("tierId"):
                    tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
                    if tier:
                        # Get category for tier
                        category = None
                        if tier.get("category_id"):
                            category_doc = db.service_categories.find_one({"_id": ObjectId(tier["category_id"])})
                            if category_doc:
                                category = ServiceCategory(
                                    id=str(category_doc["_id"]),
                                    name=category_doc["name"],
                                    description=category_doc["description"],
                                    category_type=category_doc["category_type"],
                                    image=category_doc.get("image"),
                                    is_active=category_doc["is_active"],
                                    created_at=category_doc["created_at"],
                                    updated_at=category_doc["updated_at"],
                                    tiers=[],
                                    services=[]
                                )

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
                            services=tier_service_objects,
                            category=category
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
                logger.error(f"Error processing booking {booking.get('_id')}: {e}")
                continue

        return result

    except Exception as e:
        logger.error(f"Error fetching user bookings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch bookings"
        )


@app.get("/users/me/bookings/{booking_id}", response_model=Booking)
async def get_my_booking_details(
        booking_id: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Get detailed information about a specific booking"""
    try:
        booking = db.bookings.find_one({
            "_id": ObjectId(booking_id),
            "userId": str(current_user.id)
        })

        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        # Get service and tier details (same logic as above)
        service_obj = None
        tier_obj = None

        if booking.get("serviceId"):
            service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
            if service:
                category = None
                if service.get("category_id"):
                    category_doc = db.service_categories.find_one({"_id": ObjectId(service["category_id"])})
                    if category_doc:
                        category = ServiceCategory(
                            id=str(category_doc["_id"]),
                            name=category_doc["name"],
                            description=category_doc["description"],
                            category_type=category_doc["category_type"],
                            image=category_doc.get("image"),
                            is_active=category_doc["is_active"],
                            created_at=category_doc["created_at"],
                            updated_at=category_doc["updated_at"],
                            tiers=[],
                            services=[]
                        )

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
                    updatedAt=service["updatedAt"],
                    category=category
                )

        if booking.get("tierId"):
            tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
            if tier:
                category = None
                if tier.get("category_id"):
                    category_doc = db.service_categories.find_one({"_id": ObjectId(tier["category_id"])})
                    if category_doc:
                        category = ServiceCategory(
                            id=str(category_doc["_id"]),
                            name=category_doc["name"],
                            description=category_doc["description"],
                            category_type=category_doc["category_type"],
                            image=category_doc.get("image"),
                            is_active=category_doc["is_active"],
                            created_at=category_doc["created_at"],
                            updated_at=category_doc["updated_at"],
                            tiers=[],
                            services=[]
                        )

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
                    services=tier_service_objects,
                    category=category
                )

        return Booking(
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching booking details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch booking details"
        )


@app.get("/users/me/bookings/stats", response_model=UserBookingStats)
async def get_my_booking_stats(
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Get user's booking statistics"""
    try:
        user_id = str(current_user.id)

        # Get all user bookings
        all_bookings = list(db.bookings.find({"userId": user_id}))

        # Calculate stats
        total_bookings = len(all_bookings)
        pending_bookings = len([b for b in all_bookings if b["status"] == "pending"])
        confirmed_bookings = len([b for b in all_bookings if b["status"] == "confirmed"])
        completed_bookings = len([b for b in all_bookings if b["status"] == "completed"])
        cancelled_bookings = len([b for b in all_bookings if b["status"] == "cancelled"])

        # Calculate total spent (only for confirmed/completed bookings with successful payments)
        total_spent = 0.0
        for booking in all_bookings:
            if (booking["status"] in ["confirmed", "completed"] and
                    booking.get("payment_status") == "successful" and
                    booking.get("payment_amount")):
                total_spent += booking["payment_amount"]

        # Count upcoming bookings (future bookings that are confirmed)
        now = datetime.utcnow()
        upcoming_bookings = len([
            b for b in all_bookings
            if b["status"] == "confirmed" and b["bookingDate"] > now
        ])

        return UserBookingStats(
            total_bookings=total_bookings,
            pending_bookings=pending_bookings,
            confirmed_bookings=confirmed_bookings,
            completed_bookings=completed_bookings,
            cancelled_bookings=cancelled_bookings,
            total_spent=total_spent,
            upcoming_bookings=upcoming_bookings
        )

    except Exception as e:
        logger.error(f"Error calculating booking stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate booking statistics"
        )


@app.put("/users/me/bookings/{booking_id}/cancel")
async def cancel_my_booking(
        booking_id: str,
        cancellation: BookingCancellation,
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Cancel a user's booking"""
    try:
        booking = db.bookings.find_one({
            "_id": ObjectId(booking_id),
            "userId": str(current_user.id)
        })

        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        # Check if booking can be cancelled
        if booking["status"] in ["completed", "cancelled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel a {booking['status']} booking"
            )

        # Check cancellation policy (e.g., 24 hours before booking)
        booking_date = booking["bookingDate"]
        if isinstance(booking_date, str):
            booking_date = datetime.fromisoformat(booking_date.replace('Z', '+00:00'))

        time_until_booking = booking_date - datetime.utcnow()
        if time_until_booking < timedelta(hours=24):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot cancel booking less than 24 hours before the scheduled time"
            )

        # Update booking status
        update_data = {
            "status": "cancelled",
            "updatedAt": datetime.utcnow(),
            "cancellation_reason": cancellation.reason,
            "cancelled_at": datetime.utcnow()
        }

        # If payment was made, mark for refund processing
        if (booking.get("payment_status") == "successful" and
                booking.get("payment_amount", 0) > 0):
            update_data["refund_status"] = "pending"
            update_data["refund_amount"] = booking["payment_amount"]

        result = db.bookings.update_one(
            {"_id": ObjectId(booking_id)},
            {"$set": update_data}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to cancel booking"
            )

        # Send cancellation email
        service_name = "Unknown Service"
        if booking.get("serviceId"):
            service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
            if service:
                service_name = service["name"]
        elif booking.get("tierId"):
            tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
            if tier:
                service_name = tier["name"]

        cancellation_html = f"""
        <html>
            <body>
                <h1>Booking Cancelled</h1>
                <p>Dear {current_user.firstName},</p>
                <p>Your booking for {service_name} has been successfully cancelled.</p>
                <p>Booking Details:</p>
                <ul>
                    <li>Service: {service_name}</li>
                    <li>Original Date: {booking_date.strftime("%Y-%m-%d %H:%M")}</li>
                    <li>Cancellation Date: {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}</li>
                    <li>Reason: {cancellation.reason or "Not specified"}</li>
                </ul>
                {f"<p>A refund of ₦{booking.get('payment_amount', 0):,.2f} will be processed within 5-7 business days.</p>" if booking.get('payment_amount', 0) > 0 else ""}
                <p>We're sorry to see this booking cancelled. Feel free to book again anytime.</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """

        send_email(
            current_user.email,
            "Booking Cancelled - Naija Concierge",
            cancellation_html
        )

        # Notify admin
        admin_notification = f"""
        Booking cancelled by user:
        - User: {current_user.firstName} {current_user.lastName}
        - Service: {service_name}
        - Original Date: {booking_date.strftime("%Y-%m-%d %H:%M")}
        - Reason: {cancellation.reason or "Not specified"}
        - Refund Required: {"Yes" if booking.get('payment_amount', 0) > 0 else "No"}
        - Amount: ₦{booking.get('payment_amount', 0):,.2f}
        """

        send_admin_notification("Booking Cancelled by User", admin_notification)

        return {"message": "Booking cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel booking"
        )


@app.put("/users/me/bookings/{booking_id}/reschedule")
async def reschedule_my_booking(
        booking_id: str,
        reschedule: BookingReschedule,
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Reschedule a user's booking"""
    try:
        booking = db.bookings.find_one({
            "_id": ObjectId(booking_id),
            "userId": str(current_user.id)
        })

        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        # Check if booking can be rescheduled
        if booking["status"] in ["completed", "cancelled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot reschedule a {booking['status']} booking"
            )

        # Validate new booking date
        if reschedule.new_booking_date <= datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New booking date must be in the future"
            )

        # Check rescheduling policy (e.g., 24 hours before original booking)
        original_booking_date = booking["bookingDate"]
        if isinstance(original_booking_date, str):
            original_booking_date = datetime.fromisoformat(original_booking_date.replace('Z', '+00:00'))

        time_until_original_booking = original_booking_date - datetime.utcnow()
        if time_until_original_booking < timedelta(hours=24):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot reschedule booking less than 24 hours before the original scheduled time"
            )

        # Update booking
        update_data = {
            "bookingDate": reschedule.new_booking_date,
            "updatedAt": datetime.utcnow(),
            "reschedule_reason": reschedule.reason,
            "original_booking_date": original_booking_date,
            "rescheduled_at": datetime.utcnow()
        }

        result = db.bookings.update_one(
            {"_id": ObjectId(booking_id)},
            {"$set": update_data}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to reschedule booking"
            )

        # Send reschedule confirmation email
        service_name = "Unknown Service"
        if booking.get("serviceId"):
            service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
            if service:
                service_name = service["name"]
        elif booking.get("tierId"):
            tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
            if tier:
                service_name = tier["name"]

        reschedule_html = f"""
        <html>
            <body>
                <h1>Booking Rescheduled</h1>
                <p>Dear {current_user.firstName},</p>
                <p>Your booking for {service_name} has been successfully rescheduled.</p>
                <p>Booking Details:</p>
                <ul>
                    <li>Service: {service_name}</li>
                    <li>Original Date: {original_booking_date.strftime("%Y-%m-%d %H:%M")}</li>
                    <li>New Date: {reschedule.new_booking_date.strftime("%Y-%m-%d %H:%M")}</li>
                    <li>Reason: {reschedule.reason or "Not specified"}</li>
                </ul>
                <p>We'll see you at the new scheduled time!</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """

        send_email(
            current_user.email,
            "Booking Rescheduled - Naija Concierge",
            reschedule_html
        )

        # Notify admin
        admin_notification = f"""
        Booking rescheduled by user:
        - User: {current_user.firstName} {current_user.lastName}
        - Service: {service_name}
        - Original Date: {original_booking_date.strftime("%Y-%m-%d %H:%M")}
        - New Date: {reschedule.new_booking_date.strftime("%Y-%m-%d %H:%M")}
        - Reason: {reschedule.reason or "Not specified"}
        """

        send_admin_notification("Booking Rescheduled by User", admin_notification)

        return {"message": "Booking rescheduled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rescheduling booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reschedule booking"
        )


@app.get("/users/me/bookings/upcoming", response_model=List[Booking])
async def get_my_upcoming_bookings(
        limit: int = 10,
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Get user's upcoming bookings"""
    try:
        now = datetime.utcnow()

        bookings = list(
            db.bookings.find({
                "userId": str(current_user.id),
                "bookingDate": {"$gt": now},
                "status": {"$in": ["pending", "confirmed"]}
            })
            .sort("bookingDate", 1)  # Sort by date ascending (earliest first)
            .limit(limit)
        )

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
                logger.error(f"Error processing upcoming booking {booking.get('_id')}: {e}")
                continue

        return result

    except Exception as e:
        logger.error(f"Error fetching upcoming bookings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch upcoming bookings"
        )


@app.get("/users/me/bookings/history", response_model=List[Booking])
async def get_my_booking_history(
        skip: int = 0,
        limit: int = 50,
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Get user's booking history (completed and cancelled bookings)"""
    try:
        bookings = list(
            db.bookings.find({
                "userId": str(current_user.id),
                "status": {"$in": ["completed", "cancelled"]}
            })
            .sort("updatedAt", -1)  # Sort by last update (newest first)
            .skip(skip)
            .limit(limit)
        )

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
                logger.error(f"Error processing booking history {booking.get('_id')}: {e}")
                continue

        return result

    except Exception as e:
        logger.error(f"Error fetching booking history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch booking history"
        )


@app.post("/users/me/bookings/{booking_id}/regenerate-payment")
async def regenerate_payment_url(
        booking_id: str,
        preferred_currency: str = "NGN",
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Regenerate payment URL for a booking with failed/expired payment"""
    try:
        booking = db.bookings.find_one({
            "_id": ObjectId(booking_id),
            "userId": str(current_user.id)
        })

        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        # Check if booking requires payment and payment is not successful
        if not booking.get("payment_required"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This booking does not require payment"
            )

        if booking.get("payment_status") == "successful":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Payment has already been completed for this booking"
            )

        if booking["status"] in ["completed", "cancelled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot regenerate payment for {booking['status']} booking"
            )

        # Get original amount
        original_amount = booking.get("payment_amount", 0)
        if original_amount <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid payment amount"
            )

        # Convert amount if different currency requested
        payment_amount = original_amount
        if preferred_currency != "NGN":
            try:
                payment_amount = await convert_price(original_amount, "NGN", preferred_currency)
            except Exception as e:
                logger.error(f"Currency conversion failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to convert currency"
                )

        # Generate new payment URL
        try:
            payment_url = generate_flutterwave_payment_url(
                {
                    "id": str(booking["_id"]),
                    "original_amount": original_amount
                },
                current_user.__dict__,
                payment_amount,
                preferred_currency
            )

            # Update booking with new payment URL
            db.bookings.update_one(
                {"_id": ObjectId(booking_id)},
                {"$set": {
                    "payment_url": payment_url,
                    "payment_currency": preferred_currency,
                    "payment_amount_converted": payment_amount,
                    "updatedAt": datetime.utcnow()
                }}
            )

            return {
                "payment_url": payment_url,
                "amount": payment_amount,
                "currency": preferred_currency,
                "original_amount": original_amount,
                "message": "Payment URL regenerated successfully"
            }

        except Exception as e:
            logger.error(f"Failed to generate payment URL: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate payment URL"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating payment URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate payment URL"
        )



@app.delete("/users/{user_id}")
async def delete_user(
        user_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        bookings = db.bookings.find_one({"userId": user_id})
        if bookings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete user with existing bookings",
            )

        subscriptions = db.subscriptions.find_one({"userId": user_id})
        if subscriptions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete user with existing subscriptions",
            )

        result = db.users.delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User deletion failed",
            )

        return {"message": "User deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User deletion failed",
        )


@app.post("/users/profile-image")
async def upload_profile_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_current_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/profiles")

    result = db.users.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": {"profileImage": image_url, "updatedAt": datetime.utcnow()}}
    )

    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update profile image",
        )

    return {"profileImage": image_url}

@app.get("/gallery", response_model=List[GalleryImage])
async def get_gallery(
        skip: int = 0,
        limit: int = 100,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        # current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Get all gallery images with optional filtering by category or tag.
    """
    query = {}

    if category:
        query["category"] = category
    if tag:
        query["tags"] = {"$in": [tag]}

    try:
        gallery_images = list(db.gallery.find(query).skip(skip).limit(limit))
        return [
            GalleryImage(
                id=str(img["_id"]),
                title=img["title"],
                description=img.get("description"),
                category=img["category"],
                tags=img.get("tags", []),
                image_url=img["image_url"],
                created_by=img["created_by"],
                created_at=img["created_at"],
                updated_at=img["updated_at"]
            ) for img in gallery_images
        ]
    except Exception as e:
        logger.error(f"Error fetching gallery images: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch gallery images"
        )



@app.post("/gallery", response_model=GalleryImage)
async def create_gallery_image(
        title: str = Form(...),
        description: Optional[str] = Form(None),
        category: str = Form(...),
        tags: str = Form(""),  # Comma-separated string
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Upload a new image to the gallery.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    if not title.strip() or not category.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title and category are required"
        )

    try:
        # Upload image to Cloudinary
        image_url = upload_file_to_cloudinary(file, folder="naija_concierge/gallery")

        # Process tags
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        gallery_image = GalleryImageInDB(
            title=title,
            description=description,
            category=category,
            tags=tag_list,
            image_url=image_url,
            created_by=str(current_user.id))

        result = db.gallery.insert_one(gallery_image.dict(by_alias=True))
        created_image = db.gallery.find_one({"_id": result.inserted_id})

        return GalleryImage(
            id=str(created_image["_id"]),
            title=created_image["title"],
            description=created_image.get("description"),
            category=created_image["category"],
            tags=created_image.get("tags", []),
            image_url=created_image["image_url"],
            created_by=created_image["created_by"],
            created_at=created_image["created_at"],
            updated_at=created_image["updated_at"]
        )
    except Exception as e:
        logger.error(f"Error creating gallery image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create gallery image"
        )

@app.get("/gallery/{image_id}", response_model=GalleryImage)
async def get_gallery_image(
        image_id: str,
        # current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Get a specific gallery image by ID.
    """
    try:
        image = db.gallery.find_one({"_id": ObjectId(image_id)})
        if not image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gallery image not found"
            )

        return GalleryImage(
            id=str(image["_id"]),
            title=image["title"],
            description=image.get("description"),
            category=image["category"],
            tags=image.get("tags", []),
            image_url=image["image_url"],
            created_by=image["created_by"],
            created_at=image["created_at"],
            updated_at=image["updated_at"]
        )
    except Exception as e:
        logger.error(f"Error fetching gallery image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch gallery image"
        )



@app.put("/gallery/{image_id}", response_model=GalleryImage)
async def update_gallery_image(
        image_id: str,
        image_update: GalleryImageUpdate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Update a gallery image's metadata (title, description, category, tags).
    """
    try:
        image = db.gallery.find_one({"_id": ObjectId(image_id)})
        if not image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gallery image not found"
            )

        # Check if user is admin or the creator of the image
        if current_user.role != "admin" and image["created_by"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to update this image"
            )

        update_data = image_update.dict(exclude_unset=True)
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            result = db.gallery.update_one(
                {"_id": ObjectId(image_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Gallery image update failed"
                )

        updated_image = db.gallery.find_one({"_id": ObjectId(image_id)})

        return GalleryImage(
            id=str(updated_image["_id"]),
            title=updated_image["title"],
            description=updated_image.get("description"),
            category=updated_image["category"],
            tags=updated_image.get("tags", []),
            image_url=updated_image["image_url"],
            created_by=updated_image["created_by"],
            created_at=updated_image["created_at"],
            updated_at=updated_image["updated_at"]
        )
    except Exception as e:
        logger.error(f"Error updating gallery image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update gallery image"
        )
@app.delete("/gallery/{image_id}")
async def delete_gallery_image(
        image_id: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Delete a gallery image.
    """
    try:
        image = db.gallery.find_one({"_id": ObjectId(image_id)})
        if not image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gallery image not found"
            )

        # Check if user is admin or the creator of the image
        if current_user.role != "admin" and image["created_by"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to delete this image"
            )

        # Extract public_id from the URL (alternative solution)
        if "image_url" in image:

            url_parts = image["image_url"].split('/')
            public_id_with_extension = '/'.join(url_parts[url_parts.index('upload') + 2:])
            public_id = public_id_with_extension.split('.')[0]  # Remove file extension


            cloudinary.uploader.destroy(public_id)

        result = db.gallery.delete_one({"_id": ObjectId(image_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gallery image deletion failed"
            )

        return {"message": "Gallery image deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting gallery image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete gallery image"
        )


@app.post("/gallery/{image_id}/image")
async def update_gallery_image_file(
        image_id: str,
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Update the image file for a gallery item.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    try:
        image = db.gallery.find_one({"_id": ObjectId(image_id)})
        if not image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gallery image not found"
            )

        # Check if user is admin or the creator of the image
        if current_user.role != "admin" and image["created_by"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to update this image"
            )

        # Upload new image to Cloudinary
        new_image_url = upload_file_to_cloudinary(file, folder="naija_concierge/gallery")

        # Update the image URL in the database
        result = db.gallery.update_one(
            {"_id": ObjectId(image_id)},
            {"$set": {
                "image_url": new_image_url,
                "updated_at": datetime.utcnow()
            }}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update gallery image"
            )

        return {"image_url": new_image_url}
    except Exception as e:
        logger.error(f"Error updating gallery image file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update gallery image file"
        )



# crm

@app.get("/crm/clients", response_model=List[CRMClient])
async def get_crm_clients(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        current_user: UserInDB = Depends(get_admin_user)
):
    query = {}
    if status:
        query["status"] = status

    clients = list(db.crm_clients.find(query).skip(skip).limit(limit))
    return [
        CRMClient(
            id=str(client["_id"]),
            clientName=client["clientName"],
            contactInfo=client["contactInfo"],
            serviceBooked=client["serviceBooked"],
            status=client["status"],
            assignedVendor=client.get("assignedVendor"),
            notes=client.get("notes"),
            dueDate=client.get("dueDate"),
            createdAt=client["createdAt"],
            updatedAt=client["updatedAt"]
        ) for client in clients
    ]


@app.post("/crm/clients", response_model=CRMClient)
async def create_crm_client(
        client: CRMClientCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    if not client.clientName.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client name is required"
        )
    if not client.contactInfo.get("email") and not client.contactInfo.get("phone"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one contact method (email or phone) is required"
        )

    try:
        service = db.services.find_one({"_id": ObjectId(client.serviceBooked)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found"
            )
    except Exception as e:
        logger.error(f"Invalid service ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid service ID"
        )

    client_in_db = CRMClientInDB(**client.dict())
    result = db.crm_clients.insert_one(client_in_db.dict(by_alias=True))

    created_client = db.crm_clients.find_one({"_id": result.inserted_id})

    return CRMClient(
        id=str(created_client["_id"]),
        clientName=created_client["clientName"],
        contactInfo=created_client["contactInfo"],
        serviceBooked=created_client["serviceBooked"],
        status=created_client["status"],
        assignedVendor=created_client.get("assignedVendor"),
        notes=created_client.get("notes"),
        dueDate=created_client.get("dueDate"),
        createdAt=created_client["createdAt"],
        updatedAt=created_client["updatedAt"]
    )


@app.get("/crm/clients/{client_id}", response_model=CRMClient)
async def get_crm_client(
        client_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        client = db.crm_clients.find_one({"_id": ObjectId(client_id)})
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found",
            )

        return CRMClient(
            id=str(client["_id"]),
            clientName=client["clientName"],
            contactInfo=client["contactInfo"],
            serviceBooked=client["serviceBooked"],
            status=client["status"],
            assignedVendor=client.get("assignedVendor"),
            notes=client.get("notes"),
            dueDate=client.get("dueDate"),
            createdAt=client["createdAt"],
            updatedAt=client["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error getting CRM client: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found",
        )


@app.put("/crm/clients/{client_id}", response_model=CRMClient)
async def update_crm_client(
        client_id: str,
        client_update: CRMClientUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        client = db.crm_clients.find_one({"_id": ObjectId(client_id)})
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found",
            )

        update_data = client_update.dict(exclude_unset=True)
        if update_data:
            if "clientName" in update_data and not update_data["clientName"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Client name cannot be empty"
                )
            if "serviceBooked" in update_data:
                service = db.services.find_one({"_id": ObjectId(update_data["serviceBooked"])})
                if not service:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Service not found"
                    )
            if "contactInfo" in update_data and not (update_data["contactInfo"].get("email") or update_data["contactInfo"].get("phone")):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="At least one contact method (email or phone) is required"
                )
            update_data["updatedAt"] = datetime.utcnow()
            result = db.crm_clients.update_one(
                {"_id": ObjectId(client_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Client update failed",
                )


        updated_client = db.crm_clients.find_one({"_id": ObjectId(client_id)})

        return CRMClient(
            id=str(updated_client["_id"]),
            clientName=updated_client["clientName"],
            contactInfo=updated_client["contactInfo"],
            serviceBooked=updated_client["serviceBooked"],
            status=updated_client["status"],
            assignedVendor=updated_client.get("assignedVendor"),
            notes=updated_client.get("notes"),
            dueDate=updated_client.get("dueDate"),
            createdAt=updated_client["createdAt"],
            updatedAt=updated_client["updatedAt"]
        )
    except Exception as e:
            logger.error(f"Error updating CRM client: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client update failed",
            )

@app.delete("/crm/clients/{client_id}")
async def delete_crm_client(
        client_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        client = db.crm_clients.find_one({"_id": ObjectId(client_id)})
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found",
            )

        result = db.crm_clients.delete_one({"_id": ObjectId(client_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client deletion failed",
            )

        return {"message": "Client deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting CRM client: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client deletion failed",
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


@app.put("/service-categories/{category_id}", response_model=ServiceCategory)
async def update_service_category(
        category_id: str,
        category_update: ServiceCategoryUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Update a service category"""
    try:
        category = db.service_categories.find_one({"_id": ObjectId(category_id)})
        if not category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service category not found"
            )

        update_data = category_update.dict(exclude_unset=True)
        if update_data:
            # Validate required fields if provided
            if "name" in update_data and not update_data["name"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Category name cannot be empty"
                )

            if "description" in update_data and not update_data["description"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Category description cannot be empty"
                )

            update_data["updated_at"] = datetime.utcnow()
            result = db.service_categories.update_one(
                {"_id": ObjectId(category_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Category update failed"
                )

        updated_category = db.service_categories.find_one({"_id": ObjectId(category_id)})

        # Get tiers and services for response
        tiers = []
        services = []

        if updated_category["category_type"] == ServiceCategoryType.TIERED:
            tier_docs = list(db.service_tiers.find({"category_id": category_id}))
            for tier_doc in tier_docs:
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
            id=str(updated_category["_id"]),
            name=updated_category["name"],
            description=updated_category["description"],
            category_type=updated_category["category_type"],
            image=updated_category.get("image"),
            is_active=updated_category["is_active"],
            created_at=updated_category["created_at"],
            updated_at=updated_category["updated_at"],
            tiers=tiers,
            services=services
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating service category: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update service category"
        )


@app.delete("/service-categories/{category_id}")
async def delete_service_category(
        category_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Delete a service category"""
    try:
        category = db.service_categories.find_one({"_id": ObjectId(category_id)})
        if not category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service category not found"
            )

        # Check for dependencies
        tiers_count = db.service_tiers.count_documents({"category_id": category_id})
        services_count = db.services.count_documents({"category_id": category_id})
        bookings_count = db.bookings.count_documents({"serviceId": category_id})

        if tiers_count > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete category with {tiers_count} associated tiers. Delete tiers first."
            )

        if services_count > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete category with {services_count} associated services. Delete services first."
            )

        if bookings_count > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete category with {bookings_count} associated bookings."
            )

        result = db.service_categories.delete_one({"_id": ObjectId(category_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Category deletion failed"
            )

        return {"message": "Service category deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting service category: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete service category"
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


@app.put("/service-tiers/{tier_id}", response_model=ServiceTier)
async def update_service_tier(
        tier_id: str,
        tier_update: ServiceTierUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Update a service tier"""
    try:
        tier = db.service_tiers.find_one({"_id": ObjectId(tier_id)})
        if not tier:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service tier not found"
            )

        update_data = tier_update.dict(exclude_unset=True)
        if update_data:
            # Validate required fields if provided
            if "name" in update_data and not update_data["name"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tier name cannot be empty"
                )

            if "description" in update_data and not update_data["description"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tier description cannot be empty"
                )

            if "price" in update_data and update_data["price"] < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Price cannot be negative"
                )

            # Validate category if being updated
            if "category_id" in update_data:
                category = db.service_categories.find_one({"_id": ObjectId(update_data["category_id"])})
                if not category:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Service category not found"
                    )
                if category["category_type"] != ServiceCategoryType.TIERED:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Can only assign tiers to tiered categories"
                    )

            update_data["updated_at"] = datetime.utcnow()
            result = db.service_tiers.update_one(
                {"_id": ObjectId(tier_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tier update failed"
                )

        updated_tier = db.service_tiers.find_one({"_id": ObjectId(tier_id)})

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
        if updated_tier.get("category_id"):
            category = db.service_categories.find_one({"_id": ObjectId(updated_tier["category_id"])})
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
            id=str(updated_tier["_id"]),
            name=updated_tier["name"],
            description=updated_tier["description"],
            price=updated_tier["price"],
            category_id=updated_tier["category_id"],
            image=updated_tier.get("image"),
            features=updated_tier.get("features", []),
            is_popular=updated_tier.get("is_popular", False),
            is_available=updated_tier.get("is_available", True),
            created_at=updated_tier["created_at"],
            updated_at=updated_tier["updated_at"],
            services=service_objects,
            category=category_obj
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating service tier: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update service tier"
        )


@app.delete("/service-tiers/{tier_id}")
async def delete_service_tier(
        tier_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Delete a service tier"""
    try:
        tier = db.service_tiers.find_one({"_id": ObjectId(tier_id)})
        if not tier:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service tier not found"
            )

        # Check for dependencies
        services_count = db.services.count_documents({"tier_id": tier_id})
        bookings_count = db.bookings.count_documents({"tierId": tier_id})

        if services_count > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete tier with {services_count} associated services. Delete services first."
            )

        if bookings_count > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete tier with {bookings_count} associated bookings."
            )

        result = db.service_tiers.delete_one({"_id": ObjectId(tier_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tier deletion failed"
            )

        return {"message": "Service tier deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting service tier: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete service tier"
        )


@app.put("/services/{service_id}", response_model=Service)
async def update_service(
        service_id: str,
        service_update: ServiceUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Update a service"""
    try:
        service = db.services.find_one({"_id": ObjectId(service_id)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found"
            )

        update_data = service_update.dict(exclude_unset=True)
        if update_data:
            # Validate required fields if provided
            if "name" in update_data and not update_data["name"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Service name cannot be empty"
                )

            if "description" in update_data and not update_data["description"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Service description cannot be empty"
                )

            if "duration" in update_data and not update_data["duration"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Service duration cannot be empty"
                )

            # Validate category if being updated
            if "category_id" in update_data and update_data["category_id"]:
                category = db.service_categories.find_one({"_id": ObjectId(update_data["category_id"])})
                if not category:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Service category not found"
                    )

            # Validate tier if being updated
            if "tier_id" in update_data and update_data["tier_id"]:
                tier = db.service_tiers.find_one({"_id": ObjectId(update_data["tier_id"])})
                if not tier:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Service tier not found"
                    )

                # Ensure tier belongs to the specified category
                if update_data.get("category_id") and tier["category_id"] != update_data["category_id"]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Tier does not belong to the specified category"
                    )

            update_data["updatedAt"] = datetime.utcnow()
            result = db.services.update_one(
                {"_id": ObjectId(service_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Service update failed"
                )

        updated_service = db.services.find_one({"_id": ObjectId(service_id)})

        # Get category and tier info
        category_obj = None
        tier_obj = None

        if updated_service.get("category_id"):
            category = db.service_categories.find_one({"_id": ObjectId(updated_service["category_id"])})
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

        if updated_service.get("tier_id"):
            tier = db.service_tiers.find_one({"_id": ObjectId(updated_service["tier_id"])})
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
            id=str(updated_service["_id"]),
            name=updated_service["name"],
            description=updated_service["description"],
            image=updated_service.get("image"),
            duration=updated_service["duration"],
            isAvailable=updated_service["isAvailable"],
            features=updated_service.get("features", []),
            requirements=updated_service.get("requirements", []),
            category_id=updated_service.get("category_id"),
            tier_id=updated_service.get("tier_id"),
            createdAt=updated_service["createdAt"],
            updatedAt=updated_service["updatedAt"],
            category=category_obj,
            tier=tier_obj
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update service"
        )


@app.delete("/services/{service_id}")
async def delete_service(
        service_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Delete a service"""
    try:
        service = db.services.find_one({"_id": ObjectId(service_id)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found"
            )

        # Check for dependencies
        bookings_count = db.bookings.count_documents({"serviceId": service_id})

        if bookings_count > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete service with {bookings_count} associated bookings."
            )

        result = db.services.delete_one({"_id": ObjectId(service_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Service deletion failed"
            )

        return {"message": "Service deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete service"
        )


# Bulk operations (bonus endpoints)
@app.post("/service-categories/{category_id}/toggle-status")
async def toggle_category_status(
        category_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Toggle active/inactive status of a service category"""
    try:
        category = db.service_categories.find_one({"_id": ObjectId(category_id)})
        if not category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service category not found"
            )

        new_status = not category["is_active"]
        result = db.service_categories.update_one(
            {"_id": ObjectId(category_id)},
            {"$set": {"is_active": new_status, "updated_at": datetime.utcnow()}}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status update failed"
            )

        return {
            "message": f"Category {'activated' if new_status else 'deactivated'} successfully",
            "is_active": new_status
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling category status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to toggle category status"
        )


@app.post("/service-tiers/{tier_id}/toggle-availability")
async def toggle_tier_availability(
        tier_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Toggle available/unavailable status of a service tier"""
    try:
        tier = db.service_tiers.find_one({"_id": ObjectId(tier_id)})
        if not tier:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service tier not found"
            )

        new_status = not tier["is_available"]
        result = db.service_tiers.update_one(
            {"_id": ObjectId(tier_id)},
            {"$set": {"is_available": new_status, "updated_at": datetime.utcnow()}}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Availability update failed"
            )

        return {
            "message": f"Tier {'made available' if new_status else 'made unavailable'} successfully",
            "is_available": new_status
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling tier availability: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to toggle tier availability"
        )


@app.post("/services/{service_id}/toggle-availability")
async def toggle_service_availability(
        service_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Toggle available/unavailable status of a service"""
    try:
        service = db.services.find_one({"_id": ObjectId(service_id)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found"
            )

        new_status = not service["isAvailable"]
        result = db.services.update_one(
            {"_id": ObjectId(service_id)},
            {"$set": {"isAvailable": new_status, "updatedAt": datetime.utcnow()}}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Availability update failed"
            )

        return {
            "message": f"Service {'made available' if new_status else 'made unavailable'} successfully",
            "isAvailable": new_status
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling service availability: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to toggle service availability"
        )


# Batch operations
@app.post("/service-categories/batch-update")
async def batch_update_categories(
        category_ids: List[str],
        update_data: ServiceCategoryUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Update multiple service categories at once"""
    try:
        if not category_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No category IDs provided"
            )

        # Validate all category IDs exist
        valid_ids = []
        for category_id in category_ids:
            category = db.service_categories.find_one({"_id": ObjectId(category_id)})
            if category:
                valid_ids.append(ObjectId(category_id))

        if not valid_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No valid categories found"
            )

        update_fields = update_data.dict(exclude_unset=True)
        if update_fields:
            update_fields["updated_at"] = datetime.utcnow()
            result = db.service_categories.update_many(
                {"_id": {"$in": valid_ids}},
                {"$set": update_fields}
            )

            return {
                "message": f"Updated {result.modified_count} categories successfully",
                "updated_count": result.modified_count
            }
        else:
            return {"message": "No fields to update", "updated_count": 0}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch update categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to batch update categories"
        )


@app.delete("/service-categories/batch-delete")
async def batch_delete_categories(
        category_ids: List[str],
        current_user: UserInDB = Depends(get_admin_user)
):
    """Delete multiple service categories at once"""
    try:
        if not category_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No category IDs provided"
            )

        # Check for dependencies for all categories
        for category_id in category_ids:
            tiers_count = db.service_tiers.count_documents({"category_id": category_id})
            services_count = db.services.count_documents({"category_id": category_id})
            bookings_count = db.bookings.count_documents({"serviceId": category_id})

            if tiers_count > 0 or services_count > 0 or bookings_count > 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot delete categories with existing dependencies. Category {category_id} has associated data."
                )

        # Convert to ObjectIds
        object_ids = [ObjectId(category_id) for category_id in category_ids]

        result = db.service_categories.delete_many({"_id": {"$in": object_ids}})

        return {
            "message": f"Deleted {result.deleted_count} categories successfully",
            "deleted_count": result.deleted_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch delete categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to batch delete categories"
        )


# Search and filter endpoints
@app.get("/service-categories/search")
async def search_service_categories(
        q: Optional[str] = None,
        category_type: Optional[ServiceCategoryType] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100
):
    """Search and filter service categories"""
    try:
        query = {}

        if q:
            query["$or"] = [
                {"name": {"$regex": q, "$options": "i"}},
                {"description": {"$regex": q, "$options": "i"}}
            ]

        if category_type:
            query["category_type"] = category_type

        if is_active is not None:
            query["is_active"] = is_active

        categories = list(db.service_categories.find(query).skip(skip).limit(limit))
        total_count = db.service_categories.count_documents(query)

        category_list = []
        for category in categories:
            # Get basic tier and service counts
            tiers_count = db.service_tiers.count_documents({"category_id": str(category["_id"])})
            services_count = db.services.count_documents({"category_id": str(category["_id"])})

            category_data = ServiceCategory(
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

            category_list.append({
                **category_data.dict(),
                "tiers_count": tiers_count,
                "services_count": services_count
            })

        return {
            "categories": category_list,
            "total_count": total_count,
            "page": skip // limit + 1 if limit > 0 else 1,
            "per_page": limit
        }

    except Exception as e:
        logger.error(f"Error searching service categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search service categories"
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



@app.get("/service-tiers/{tier_id}/convert")
async def convert_tier_price(
    tier_id: str,
    currency: str,
    # current_user: UserInDB = Depends(get_current_active_user)
):
    """Convert tier price to the specified currency"""
    allowed_currencies = ["NGN", "USD", "EUR", "GBP"]
    if currency not in allowed_currencies:
        raise HTTPException(
            status_code=400,
            detail=f"Currency must be one of {allowed_currencies}"
        )

    try:
        tier = db.service_tiers.find_one({"_id": ObjectId(tier_id)})
        if not tier:
            raise HTTPException(
                status_code=404,
                detail="Service tier not found"
            )
    except Exception as e:
        logger.error(f"Error fetching tier: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid tier ID"
        )

    base_currency = "NGN"
    amount = tier["price"]

    if currency != base_currency:
        try:
            amount = await convert_price(tier["price"], base_currency, currency)
        except Exception as e:
            logger.error(f"Currency conversion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to convert currency"
            )

    return {
        "convertedPrice": amount,
        "originalPrice": tier["price"],
        "originalCurrency": base_currency,
        "targetCurrency": currency,
        "exchangeRate": amount / tier["price"] if tier["price"] > 0 else 0
    }



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


@app.post("/bookings/tier", response_model=Booking)
async def create_tier_booking_with_currency(
        tier_id: str,
        booking_date: datetime,
        preferred_currency: str = "NGN",
        special_requests: Optional[str] = None,
        current_user: UserInDB = Depends(get_current_active_user)
):
    """Create a tier booking with multi-currency support"""

    # Validate currency
    allowed_currencies = ["NGN", "USD", "EUR", "GBP"]
    if preferred_currency not in allowed_currencies:
        raise HTTPException(
            status_code=400,
            detail=f"Currency must be one of {allowed_currencies}"
        )

    # Get tier details
    try:
        tier = db.service_tiers.find_one({"_id": ObjectId(tier_id)})
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
    except Exception as e:
        logger.error(f"Error checking tier: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid tier ID"
        )

    # Convert price to preferred currency
    original_price = tier["price"]  # Assuming stored in NGN
    converted_price = original_price

    if preferred_currency != "NGN":
        try:
            converted_price = await convert_price(original_price, "NGN", preferred_currency)
        except Exception as e:
            logger.error(f"Currency conversion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to convert currency"
            )

    # Create booking
    booking_data = BookingInDB(
        userId=str(current_user.id),
        tierId=tier_id,
        bookingDate=booking_date,
        status="pending",
        specialRequests=special_requests,
        booking_type=BookingType.TIER_BOOKING,
        payment_required=True,
        payment_amount=converted_price,
        payment_status=PaymentStatus.PENDING
    )

    result = db.bookings.insert_one(booking_data.dict(by_alias=True))
    created_booking = db.bookings.find_one({"_id": result.inserted_id})

    # Generate payment URL with proper currency
    try:
        payment_url = generate_flutterwave_payment_url(
            {
                "id": str(created_booking["_id"]),
                "original_amount": original_price
            },
            current_user.__dict__,
            converted_price,
            preferred_currency
        )

        # Update booking with payment URL
        db.bookings.update_one(
            {"_id": result.inserted_id},
            {"$set": {"payment_url": payment_url}}
        )

        created_booking["payment_url"] = payment_url

    except Exception as e:
        logger.error(f"Payment URL generation failed: {e}")
        # Continue without payment URL

    # Send confirmation email
    tier_services = list(db.services.find({"tier_id": tier_id}))
    service_list = ", ".join([s["name"] for s in tier_services])

    booking_html = f"""
    <html>
        <body>
            <h1>Tier Booking Confirmation - Payment Required</h1>
            <p>Dear {current_user.firstName},</p>
            <p>Your booking for {tier["name"]} has been received.</p>
            <p>To confirm your booking, please complete the payment.</p>
            <p>Booking Details:</p>
            <ul>
                <li>Tier: {tier["name"]}</li>
                <li>Services: {service_list}</li>
                <li>Date: {booking_date.strftime("%Y-%m-%d %H:%M")}</li>
                <li>Amount: {preferred_currency} {converted_price:,.2f}</li>
                <li>Original Price: NGN {original_price:,.2f}</li>
                <li>Status: Pending Payment</li>
            </ul>
            <p>Payment URL: <a href='{payment_url}'>Complete Payment</a></p>
            <p>Best regards,<br>The Naija Concierge Team</p>
        </body>
    </html>
    """
    send_email(current_user.email, "Tier Booking Confirmation - Payment Required", booking_html)

    # Get tier object for response
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
        price=converted_price,  # Return converted price
        category_id=tier["category_id"],
        image=tier.get("image"),
        features=tier.get("features", []),
        is_popular=tier.get("is_popular", False),
        is_available=tier.get("is_available", True),
        created_at=tier["created_at"],
        updated_at=tier["updated_at"],
        services=tier_service_objects
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
        service=None,
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




@app.get("/admin/bookings", response_model=List[Booking])
async def admin_get_all_bookings(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = Query(None),
        booking_type: Optional[BookingType] = Query(None),
        payment_status: Optional[PaymentStatus] = Query(None),
        user_email: Optional[str] = Query(None),
        date_from: Optional[datetime] = Query(None),
        date_to: Optional[datetime] = Query(None),
        search: Optional[str] = Query(None),
        sort_by: Optional[str] = Query("createdAt"),
        sort_order: Optional[str] = Query("desc"),
        current_user: UserInDB = Depends(get_admin_user)
):
    """Get all bookings with advanced filtering for admin"""
    try:
        query = {}

        # Apply filters
        if status:
            query["status"] = status
        if booking_type:
            query["booking_type"] = booking_type
        if payment_status:
            query["payment_status"] = payment_status
        if user_email:
            user = db.users.find_one({"email": {"$regex": user_email, "$options": "i"}})
            if user:
                query["userId"] = str(user["_id"])
            else:
                return []  # No user found with that email

        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["$gte"] = date_from
            if date_to:
                date_filter["$lte"] = date_to
            query["bookingDate"] = date_filter

        # Search across multiple fields
        if search:
            users = list(db.users.find({
                "$or": [
                    {"firstName": {"$regex": search, "$options": "i"}},
                    {"lastName": {"$regex": search, "$options": "i"}},
                    {"email": {"$regex": search, "$options": "i"}}
                ]
            }))
            user_ids = [str(user["_id"]) for user in users]

            search_conditions = [
                {"userId": {"$in": user_ids}},
                {"specialRequests": {"$regex": search, "$options": "i"}},
                {"admin_notes": {"$regex": search, "$options": "i"}},
                {"payment_reference": {"$regex": search, "$options": "i"}}
            ]

            if "$or" in query:
                query["$and"] = [{"$or": query["$or"]}, {"$or": search_conditions}]
                del query["$or"]
            else:
                query["$or"] = search_conditions

        # Sort configuration
        sort_direction = -1 if sort_order.lower() == "desc" else 1
        sort_field = sort_by if sort_by in ["createdAt", "bookingDate", "updatedAt", "status"] else "createdAt"

        bookings = list(
            db.bookings.find(query)
            .sort(sort_field, sort_direction)
            .skip(skip)
            .limit(limit)
        )

        result = []
        for booking in bookings:
            try:
                # Get user info
                user = get_user_by_id(booking["userId"])

                # Get service/tier details (same logic as existing endpoints)
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
                        tier_service_objects = []
                        tier_services = list(db.services.find({"tier_id": booking["tierId"]}))
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

                booking_obj = Booking(
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

                # Add user info to the response (extend the model if needed)
                booking_dict = booking_obj.dict()
                if user:
                    booking_dict["user_info"] = {
                        "firstName": user.firstName,
                        "lastName": user.lastName,
                        "email": user.email,
                        "phone": user.phone
                    }
                booking_dict["admin_notes"] = booking.get("admin_notes", "")
                booking_dict["assigned_to"] = booking.get("assigned_to", "")
                booking_dict["priority"] = booking.get("priority", "medium")

                result.append(booking_dict)

            except Exception as e:
                logger.error(f"Error processing booking {booking.get('_id')}: {e}")
                continue

        return result

    except Exception as e:
        logger.error(f"Error fetching admin bookings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch bookings"
        )


@app.put("/admin/bookings/{booking_id}")
async def admin_update_booking(
        booking_id: str,
        booking_update: AdminBookingUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Update any booking as admin"""
    try:
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        update_data = booking_update.dict(exclude_unset=True)
        if update_data:
            update_data["updatedAt"] = datetime.utcnow()
            update_data["last_updated_by"] = str(current_user.id)

            # Track status changes
            if "status" in update_data and update_data["status"] != booking["status"]:
                update_data["status_history"] = booking.get("status_history", [])
                update_data["status_history"].append({
                    "previous_status": booking["status"],
                    "new_status": update_data["status"],
                    "changed_by": str(current_user.id),
                    "changed_at": datetime.utcnow(),
                    "notes": update_data.get("admin_notes", "")
                })

            result = db.bookings.update_one(
                {"_id": ObjectId(booking_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Booking update failed"
                )

            # Get user and send notification if status changed
            if "status" in update_data:
                user = get_user_by_id(booking["userId"])
                if user:
                    service_name = "Service"
                    if booking.get("serviceId"):
                        service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                        if service:
                            service_name = service["name"]
                    elif booking.get("tierId"):
                        tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
                        if tier:
                            service_name = tier["name"]

                    status_update_html = f"""
                    <html>
                        <body>
                            <h1>Booking Status Update</h1>
                            <p>Dear {user.firstName},</p>
                            <p>Your booking for {service_name} has been updated.</p>
                            <p>New Status: <strong>{update_data['status'].title()}</strong></p>
                            <p>Booking Date: {booking['bookingDate'].strftime('%Y-%m-%d %H:%M')}</p>
                            {f"<p>Notes: {update_data.get('admin_notes', '')}</p>" if update_data.get('admin_notes') else ""}
                            <p>If you have any questions, please contact us.</p>
                            <p>Best regards,<br>The Naija Concierge Team</p>
                        </body>
                    </html>
                    """
                    send_email(user.email, f"Booking Status Update - {service_name}", status_update_html)

        return {"message": "Booking updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update booking"
        )


@app.delete("/admin/bookings/{booking_id}")
async def admin_delete_booking(
        booking_id: str,
        reason: Optional[str] = None,
        notify_user: bool = True,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Delete any booking as admin"""
    try:
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        # Get user info before deletion for notification
        user = get_user_by_id(booking["userId"])

        # Get service/tier name for notification
        service_name = "Service"
        if booking.get("serviceId"):
            service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
            if service:
                service_name = service["name"]
        elif booking.get("tierId"):
            tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
            if tier:
                service_name = tier["name"]

        # Archive the booking before deletion (optional)
        archive_data = {
            **booking,
            "deleted_at": datetime.utcnow(),
            "deleted_by": str(current_user.id),
            "deletion_reason": reason
        }
        db.archived_bookings.insert_one(archive_data)

        # Delete the booking
        result = db.bookings.delete_one({"_id": ObjectId(booking_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Booking deletion failed"
            )

        # Notify user if requested
        if notify_user and user:
            deletion_html = f"""
            <html>
                <body>
                    <h1>Booking Cancelled</h1>
                    <p>Dear {user.firstName},</p>
                    <p>Your booking for {service_name} has been cancelled by our team.</p>
                    <p>Booking Details:</p>
                    <ul>
                        <li>Service: {service_name}</li>
                        <li>Original Date: {booking['bookingDate'].strftime('%Y-%m-%d %H:%M')}</li>
                        <li>Reason: {reason or 'Administrative cancellation'}</li>
                    </ul>
                    <p>If you have any questions or would like to reschedule, please contact us.</p>
                    <p>Best regards,<br>The Naija Concierge Team</p>
                </body>
            </html>
            """
            send_email(user.email, f"Booking Cancelled - {service_name}", deletion_html)

        return {"message": "Booking deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete booking"
        )


@app.put("/admin/bookings/{booking_id}/status")
async def admin_update_booking_status(
        booking_id: str,
        status_update: BookingStatusUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Update booking status with notifications"""
    try:
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        # Validate status
        valid_statuses = ["pending", "confirmed", "completed", "cancelled", "in_progress", "on_hold"]
        if status_update.status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )

        update_data = {
            "status": status_update.status,
            "updatedAt": datetime.utcnow(),
            "last_updated_by": str(current_user.id)
        }

        if status_update.admin_notes:
            update_data["admin_notes"] = status_update.admin_notes

        # Add to status history
        status_history = booking.get("status_history", [])
        status_history.append({
            "previous_status": booking["status"],
            "new_status": status_update.status,
            "changed_by": str(current_user.id),
            "changed_at": datetime.utcnow(),
            "notes": status_update.admin_notes or ""
        })
        update_data["status_history"] = status_history

        result = db.bookings.update_one(
            {"_id": ObjectId(booking_id)},
            {"$set": update_data}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status update failed"
            )

        # Send notification to user if requested
        if status_update.notify_user:
            user = get_user_by_id(booking["userId"])
            if user:
                service_name = "Service"
                if booking.get("serviceId"):
                    service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                    if service:
                        service_name = service["name"]
                elif booking.get("tierId"):
                    tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
                    if tier:
                        service_name = tier["name"]

                status_html = f"""
                <html>
                    <body>
                        <h1>Booking Status Update</h1>
                        <p>Dear {user.firstName},</p>
                        <p>Your booking for {service_name} has been updated to: <strong>{status_update.status.title()}</strong></p>
                        <p>Booking Date: {booking['bookingDate'].strftime('%Y-%m-%d %H:%M')}</p>
                        {f"<p>Update Notes: {status_update.admin_notes}</p>" if status_update.admin_notes else ""}
                        <p>Best regards,<br>The Naija Concierge Team</p>
                    </body>
                </html>
                """
                send_email(user.email, f"Booking Status Update - {service_name}", status_html)

        return {"message": f"Booking status updated to {status_update.status}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating booking status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update booking status"
        )


@app.put("/admin/bookings/{booking_id}/assign")
async def admin_assign_booking(
        booking_id: str,
        assignment: BookingAssignment,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Assign booking to staff member or vendor"""
    try:
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        update_data = {
            "assigned_to": assignment.assigned_to,
            "assignment_notes": assignment.assignment_notes,
            "updatedAt": datetime.utcnow(),
            "assigned_by": str(current_user.id),
            "assigned_at": datetime.utcnow()
        }

        result = db.bookings.update_one(
            {"_id": ObjectId(booking_id)},
            {"$set": update_data}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assignment failed"
            )

        # Log assignment history
        assignment_log = {
            "booking_id": booking_id,
            "assigned_to": assignment.assigned_to,
            "assigned_by": str(current_user.id),
            "assigned_at": datetime.utcnow(),
            "notes": assignment.assignment_notes
        }
        db.assignment_history.insert_one(assignment_log)

        return {"message": f"Booking assigned to {assignment.assigned_to}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign booking"
        )


@app.get("/admin/bookings/stats")
async def admin_get_booking_stats(
        date_from: Optional[datetime] = Query(None),
        date_to: Optional[datetime] = Query(None),
        current_user: UserInDB = Depends(get_admin_user)
):
    """Get comprehensive booking statistics for admin dashboard"""
    try:
        # Build date filter
        date_filter = {}
        if date_from:
            date_filter["$gte"] = date_from
        if date_to:
            date_filter["$lte"] = date_to

        query = {}
        if date_filter:
            query["createdAt"] = date_filter

        # Get all bookings in date range
        all_bookings = list(db.bookings.find(query))

        # Basic counts
        total_bookings = len(all_bookings)
        pending_bookings = len([b for b in all_bookings if b["status"] == "pending"])
        confirmed_bookings = len([b for b in all_bookings if b["status"] == "confirmed"])
        completed_bookings = len([b for b in all_bookings if b["status"] == "completed"])
        cancelled_bookings = len([b for b in all_bookings if b["status"] == "cancelled"])

        # Revenue calculations
        total_revenue = sum(b.get("payment_amount", 0) for b in all_bookings
                            if b.get("payment_status") == "successful")

        pending_revenue = sum(b.get("payment_amount", 0) for b in all_bookings
                              if b.get("payment_status") == "pending" and b.get("payment_required"))

        # Booking type breakdown
        consultation_bookings = len([b for b in all_bookings if b.get("booking_type") == "consultation"])
        tier_bookings = len([b for b in all_bookings if b.get("booking_type") == "tier_booking"])

        # Payment status breakdown
        successful_payments = len([b for b in all_bookings if b.get("payment_status") == "successful"])
        pending_payments = len([b for b in all_bookings if b.get("payment_status") == "pending"])
        failed_payments = len([b for b in all_bookings if b.get("payment_status") == "failed"])

        # Top services/tiers
        service_counts = {}
        tier_counts = {}

        for booking in all_bookings:
            if booking.get("serviceId"):
                service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                if service:
                    service_name = service["name"]
                    service_counts[service_name] = service_counts.get(service_name, 0) + 1

            if booking.get("tierId"):
                tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
                if tier:
                    tier_name = tier["name"]
                    tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1

        # Sort and get top 5
        top_services = sorted(service_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_tiers = sorted(tier_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Average booking value
        paid_bookings = [b for b in all_bookings if b.get("payment_amount", 0) > 0]
        avg_booking_value = (
                    sum(b["payment_amount"] for b in paid_bookings) / len(paid_bookings)) if paid_bookings else 0

        # Conversion rate (confirmed + completed vs total)
        conversion_rate = (
                    (confirmed_bookings + completed_bookings) / total_bookings * 100) if total_bookings > 0 else 0

        return {
            "total_bookings": total_bookings,
            "pending_bookings": pending_bookings,
            "confirmed_bookings": confirmed_bookings,
            "completed_bookings": completed_bookings,
            "cancelled_bookings": cancelled_bookings,
            "total_revenue": total_revenue,
            "pending_revenue": pending_revenue,
            "consultation_bookings": consultation_bookings,
            "tier_bookings": tier_bookings,
            "successful_payments": successful_payments,
            "pending_payments": pending_payments,
            "failed_payments": failed_payments,
            "top_services": [{"name": name, "count": count} for name, count in top_services],
            "top_tiers": [{"name": name, "count": count} for name, count in top_tiers],
            "average_booking_value": round(avg_booking_value, 2),
            "conversion_rate": round(conversion_rate, 2),
            "date_range": {
                "from": date_from.isoformat() if date_from else None,
                "to": date_to.isoformat() if date_to else None
            }
        }

    except Exception as e:
        logger.error(f"Error getting booking stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get booking statistics"
        )


@app.post("/admin/bookings/bulk-update")
async def admin_bulk_update_bookings(
        bulk_action: BulkBookingAction,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Perform bulk operations on multiple bookings"""
    try:
        if not bulk_action.booking_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No booking IDs provided"
            )

        # Validate all booking IDs exist
        valid_ids = []
        for booking_id in bulk_action.booking_ids:
            if db.bookings.find_one({"_id": ObjectId(booking_id)}):
                valid_ids.append(ObjectId(booking_id))

        if not valid_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No valid bookings found"
            )

        updated_count = 0

        if bulk_action.action == "update_status":
            new_status = bulk_action.data.get("status")
            if not new_status:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Status is required for update_status action"
                )

            update_data = {
                "status": new_status,
                "updatedAt": datetime.utcnow(),
                "last_updated_by": str(current_user.id)
            }

            if bulk_action.data.get("admin_notes"):
                update_data["admin_notes"] = bulk_action.data["admin_notes"]

            result = db.bookings.update_many(
                {"_id": {"$in": valid_ids}},
                {"$set": update_data}
            )
            updated_count = result.modified_count

        elif bulk_action.action == "assign":
            assigned_to = bulk_action.data.get("assigned_to")
            if not assigned_to:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="assigned_to is required for assign action"
                )

            update_data = {
                "assigned_to": assigned_to,
                "updatedAt": datetime.utcnow(),
                "assigned_by": str(current_user.id),
                "assigned_at": datetime.utcnow()
            }

            if bulk_action.data.get("assignment_notes"):
                update_data["assignment_notes"] = bulk_action.data["assignment_notes"]

            result = db.bookings.update_many(
                {"_id": {"$in": valid_ids}},
                {"$set": update_data}
            )
            updated_count = result.modified_count

        elif bulk_action.action == "delete":
            # Archive before deleting
            bookings_to_archive = list(db.bookings.find({"_id": {"$in": valid_ids}}))
            for booking in bookings_to_archive:
                archive_data = {
                    **booking,
                    "deleted_at": datetime.utcnow(),
                    "deleted_by": str(current_user.id),
                    "deletion_reason": bulk_action.data.get("reason", "Bulk deletion")
                }
                db.archived_bookings.insert_one(archive_data)

            result = db.bookings.delete_many({"_id": {"$in": valid_ids}})
            updated_count = result.deleted_count

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid action. Supported actions: update_status, assign, delete"
            )

        return {
            "message": f"Bulk {bulk_action.action} completed",
            "updated_count": updated_count,
            "total_requested": len(bulk_action.booking_ids)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk booking operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform bulk operation"
        )


@app.post("/admin/bookings/{booking_id}/notes")
async def admin_add_booking_note(
        booking_id: str,
        note: BookingAdminNote,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Add admin note to booking"""
    try:
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        new_note = {
            "note": note.note,
            "note_type": note.note_type,
            "added_by": str(current_user.id),
            "added_at": datetime.utcnow()
        }

        admin_notes = booking.get("admin_notes_history", [])
        admin_notes.append(new_note)

        result = db.bookings.update_one(
            {"_id": ObjectId(booking_id)},
            {"$set": {
                "admin_notes": note.note,  # Keep the latest note in main field
                "admin_notes_history": admin_notes,
                "updatedAt": datetime.utcnow()
            }}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to add note"
            )

        return {"message": "Note added successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding booking note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add note"
        )


@app.get("/admin/bookings/export")
async def admin_export_bookings(
        format: str = Query("csv", regex="^(csv|json)$"),
        date_from: Optional[datetime] = Query(None),
        date_to: Optional[datetime] = Query(None),
        status: Optional[str] = Query(None),
        current_user: UserInDB = Depends(get_admin_user)
):
    """Export bookings data in CSV or JSON format"""
    try:
        query = {}

        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["$gte"] = date_from
            if date_to:
                date_filter["$lte"] = date_to
            query["createdAt"] = date_filter

        if status:
            query["status"] = status

        bookings = list(db.bookings.find(query))

        export_data = []
        for booking in bookings:
            # Get user info
            user = get_user_by_id(booking["userId"])

            # Get service/tier info
            service_name = ""
            service_type = ""
            if booking.get("serviceId"):
                service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                if service:
                    service_name = service["name"]
                    service_type = "Individual Service"
            elif booking.get("tierId"):
                tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
                if tier:
                    service_name = tier["name"]
                    service_type = "Tier Booking"

            export_record = {
                "booking_id": str(booking["_id"]),
                "user_name": f"{user.firstName} {user.lastName}" if user else "Unknown",
                "user_email": user.email if user else "Unknown",
                "user_phone": user.phone if user else "",
                "service_name": service_name,
                "service_type": service_type,
                "booking_date": booking["bookingDate"].isoformat(),
                "status": booking["status"],
                "booking_type": booking.get("booking_type", ""),
                "payment_required": booking.get("payment_required", False),
                "payment_amount": booking.get("payment_amount", 0),
                "payment_status": booking.get("payment_status", ""),
                "special_requests": booking.get("specialRequests", ""),
                "admin_notes": booking.get("admin_notes", ""),
                "assigned_to": booking.get("assigned_to", ""),
                "created_at": booking["createdAt"].isoformat(),
                "updated_at": booking["updatedAt"].isoformat()
            }
            export_data.append(export_record)

        if format == "json":
            return {
                "data": export_data,
                "total_records": len(export_data),
                "exported_at": datetime.utcnow().isoformat()
            }

        # For CSV format, you would typically return a file response
        # This is a simplified version that returns CSV-like data
        return {
            "message": "CSV export functionality would be implemented here",
            "records_count": len(export_data),
            "sample_data": export_data[:5] if export_data else []
        }

    except Exception as e:
        logger.error(f"Error exporting bookings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export bookings"
        )


@app.get("/admin/bookings/{booking_id}/history")
async def admin_get_booking_history(
        booking_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    """Get detailed history of a booking including all changes"""
    try:
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found"
            )

        # Get status history
        status_history = booking.get("status_history", [])

        # Get notes history
        notes_history = booking.get("admin_notes_history", [])

        # Get assignment history
        assignment_history = list(db.assignment_history.find({"booking_id": booking_id}))

        # Combine and sort all history by timestamp
        all_history = []

        for entry in status_history:
            all_history.append({
                "type": "status_change",
                "timestamp": entry["changed_at"],
                "data": entry
            })

        for entry in notes_history:
            all_history.append({
                "type": "note_added",
                "timestamp": entry["added_at"],
                "data": entry
            })

        for entry in assignment_history:
            all_history.append({
                "type": "assignment",
                "timestamp": entry["assigned_at"],
                "data": entry
            })

        # Sort by timestamp (newest first)
        all_history.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "booking_id": booking_id,
            "current_status": booking["status"],
            "created_at": booking["createdAt"],
            "last_updated": booking["updatedAt"],
            "history": all_history
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting booking history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get booking history"
        )




# Payment webhook endpoint
# @app.post("/webhooks/flutterwave")
# async def flutterwave_webhook(request: Request):
#     """Handle Flutterwave payment webhooks"""
#     try:
#         # Get raw body for signature verification
#         body = await request.body()
#         signature = request.headers.get("verif-hash")
#
#         if not verify_webhook_signature(body.decode(), signature):
#             logger.warning("Invalid webhook signature")
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid signature"
#             )
#
#         # Parse webhook data
#         webhook_data = json.loads(body.decode())
#         event = webhook_data.get("event")
#         data = webhook_data.get("data", {})
#
#         if event == "charge.completed":
#             tx_ref = data.get("tx_ref")
#             flw_ref = data.get("flw_ref")
#             payment_status = data.get("status")
#             amount = data.get("amount")
#
#             if not tx_ref:
#                 logger.error("No transaction reference in webhook")
#                 return {"status": "error", "message": "No transaction reference"}
#
#             # Find booking by transaction reference
#             booking = db.bookings.find_one({"flutterwave_tx_ref": tx_ref})
#             if not booking:
#                 logger.error(f"No booking found for tx_ref: {tx_ref}")
#                 return {"status": "error", "message": "Booking not found"}
#
#             # Verify payment with Flutterwave
#             try:
#                 verification_response = verify_flutterwave_payment(tx_ref)
#                 verified_data = verification_response.get("data", {})
#                 verified_status = verified_data.get("status")
#                 verified_amount = verified_data.get("amount")
#
#                 if verified_status == "successful" and verified_amount == booking.get("payment_amount"):
#                     # Update booking status
#                     update_data = {
#                         "payment_status": PaymentStatus.SUCCESSFUL,
#                         "payment_reference": flw_ref,
#                         "status": "confirmed",
#                         "updatedAt": datetime.utcnow()
#                     }
#
#                     db.bookings.update_one(
#                         {"_id": ObjectId(booking["_id"])},
#                         {"$set": update_data}
#                     )
#
#                     # Send confirmation email
#                     user = get_user_by_id(booking["userId"])
#                     if user:
#                         # Get tier/service details
#                         tier_name = "Unknown"
#                         if booking.get("tierId"):
#                             tier = db.service_tiers.find_one({"_id": ObjectId(booking["tierId"])})
#                             if tier:
#                                 tier_name = tier["name"]
#
#                         confirmation_html = f"""
#                         <html>
#                             <body>
#                                 <h1>Payment Successful - Booking Confirmed</h1>
#                                 <p>Dear {user.firstName},</p>
#                                 <p>Your payment has been successfully processed and your booking is now confirmed.</p>
#                                 <p>Payment Details:</p>
#                                 <ul>
#                                     <li>Tier: {tier_name}</li>
#                                     <li>Amount: ₦{verified_amount}</li>
#                                     <li>Reference: {flw_ref}</li>
#                                     <li>Status: Confirmed</li>
#                                 </ul>
#                                 <p>Our team will contact you shortly to coordinate the service delivery.</p>
#                                 <p>Best regards,<br>The Naija Concierge Team</p>
#                             </body>
#                         </html>
#                         """
#                         send_email(user.email, "Payment Successful - Booking Confirmed", confirmation_html)
#
#                     # Send admin notification
#                     admin_notification = f"""
#                     Payment received for booking:
#                     - Booking ID: {booking["_id"]}
#                     - Client: {user.firstName} {user.lastName} if user else "Unknown"
#                     - Tier: {tier_name}
#                     - Amount: ₦{verified_amount}
#                     - Reference: {flw_ref}
#                     - Status: Confirmed
#                     """
#                     send_admin_notification("Payment Received", admin_notification)
#
#                     logger.info(f"Payment successful for booking {booking['_id']}")
#
#                 elif verified_status == "failed":
#                     # Update booking as failed
#                     db.bookings.update_one(
#                         {"_id": ObjectId(booking["_id"])},
#                         {"$set": {
#                             "payment_status": PaymentStatus.FAILED,
#                             "payment_reference": flw_ref,
#                             "updatedAt": datetime.utcnow()
#                         }}
#                     )
#
#                     logger.info(f"Payment failed for booking {booking['_id']}")
#
#                 else:
#                     logger.warning(f"Unhandled payment status: {verified_status}")
#
#             except Exception as e:
#                 logger.error(f"Error verifying payment: {e}")
#                 return {"status": "error", "message": "Payment verification failed"}
#
#         return {"status": "success"}
#
#     except Exception as e:
#         logger.error(f"Error processing webhook: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Webhook processing failed"
#         )


@app.post("/webhooks/flutterwave")
async def flutterwave_webhook_enhanced(request: Request):
    """Enhanced webhook handler with multi-currency support"""
    try:
        body = await request.body()
        signature = request.headers.get("verif-hash")

        if not verify_webhook_signature(body.decode(), signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid signature"
            )

        webhook_data = json.loads(body.decode())
        event = webhook_data.get("event")
        data = webhook_data.get("data", {})

        if event == "charge.completed":
            tx_ref = data.get("tx_ref")
            flw_ref = data.get("flw_ref")
            payment_status = data.get("status")
            paid_amount = data.get("amount")
            payment_currency = data.get("currency")

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
                verified_currency = verified_data.get("currency")

                # Check if payment is successful and amounts match
                expected_amount = booking.get("payment_amount", 0)
                amount_matches = abs(float(verified_amount) - float(expected_amount)) < 0.01

                if verified_status == "successful" and amount_matches:
                    # Update booking status
                    update_data = {
                        "payment_status": PaymentStatus.SUCCESSFUL,
                        "payment_reference": flw_ref,
                        "status": "confirmed",
                        "payment_currency_final": verified_currency,
                        "payment_amount_final": verified_amount,
                        "updatedAt": datetime.utcnow()
                    }

                    db.bookings.update_one(
                        {"_id": ObjectId(booking["_id"])},
                        {"$set": update_data}
                    )

                    # Send confirmation email
                    user = get_user_by_id(booking["userId"])
                    if user:
                        # Get tier details
                        tier_name = "Unknown"
                        original_amount = booking.get("payment_amount_original", verified_amount)

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
                                    <li>Amount Paid: {verified_currency} {verified_amount}</li>
                                    <li>Original Price: NGN {original_amount}</li>
                                    <li>Reference: {flw_ref}</li>
                                    <li>Status: Confirmed</li>
                                </ul>
                                <p>Our team will contact you shortly to coordinate the service delivery.</p>
                                <p>Best regards,<br>The Naija Concierge Team</p>
                            </body>
                        </html>
                        """
                        send_email(user.email, "Payment Successful - Booking Confirmed", confirmation_html)

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
                    logger.warning(
                        f"Payment verification failed - Status: {verified_status}, Amount match: {amount_matches}")

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


@app.get("/exchange-rates")
async def get_current_exchange_rates():
    """Get current exchange rates for all supported currencies"""
    base_currency = "NGN"
    target_currencies = ["USD", "EUR", "GBP"]

    rates = {"NGN": 1.0}  # Base currency

    for currency in target_currencies:
        try:
            rate = await get_exchange_rate(base_currency, currency)
            rates[currency] = rate
        except Exception as e:
            logger.error(f"Failed to get rate for {currency}: {e}")
            # Use fallback rates
            fallback_rates = {"USD": 0.00065, "EUR": 0.0006, "GBP": 0.00052}
            rates[currency] = fallback_rates.get(currency, 1.0)

    return {
        "base_currency": base_currency,
        "rates": rates,
        "timestamp": datetime.utcnow().isoformat()
    }

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


# Blog Routes
@app.get("/blogs", response_model=List[Blog])
async def get_blogs(
        skip: int = 0,
        limit: int = 100,
        tag: Optional[str] = None
):
    query = {}
    if tag:
        query["tags"] = {"$in": [tag]}

    blogs = list(db.blogs.find(query).skip(skip).limit(limit))
    return [
        Blog(
            id=str(blog["_id"]),
            title=blog["title"],
            slug=blog["slug"],
            content=blog["content"],
            excerpt=blog["excerpt"],
            coverImage=blog.get("coverImage"),
            author=blog["author"],
            tags=blog["tags"],
            createdAt=blog["createdAt"],
            updatedAt=blog["updatedAt"]
        ) for blog in blogs
    ]


@app.get("/blogs/{blog_id}", response_model=Blog)
async def get_blog(blog_id: str):
    try:
        blog = db.blogs.find_one({"_id": ObjectId(blog_id)})
        if not blog:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Blog not found",
            )

        return Blog(
            id=str(blog["_id"]),
            title=blog["title"],
            slug=blog["slug"],
            content=blog["content"],
            excerpt=blog["excerpt"],
            coverImage=blog.get("coverImage"),
            author=blog["author"],
            tags=blog["tags"],
            createdAt=blog["createdAt"],
            updatedAt=blog["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error getting blog: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Blog not found",
        )

@app.get("/blogs/blog/{slug}", response_model=Blog)
async def get_blog_by_slug(slug: str):
    blog = db.blogs.find_one({"slug": slug})
    if not blog:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Blog not found",
        )

    return Blog(
        id=str(blog["_id"]),
        title=blog["title"],
        slug=blog["slug"],
        content=blog["content"],
        excerpt=blog["excerpt"],
        coverImage=blog.get("coverImage"),
        author=blog["author"],
        tags=blog["tags"],
        createdAt=blog["createdAt"],
        updatedAt=blog["updatedAt"]
    )


@app.post("/blogs", response_model=Blog)
async def create_blog(
        blog: BlogCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    if not blog.title.strip() or not blog.slug.strip() or not blog.content.strip() or not blog.excerpt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title, slug, content, and excerpt are required"
        )
    if not blog.author.get("name"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Author name is required"
        )

    existing_blog = db.blogs.find_one({"slug": blog.slug})
    if existing_blog:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Blog with this slug already exists",
        )

    blog_in_db = BlogInDB(**blog.dict())
    result = db.blogs.insert_one(blog_in_db.dict(by_alias=True))

    created_blog = db.blogs.find_one({"_id": result.inserted_id})

    return Blog(
        id=str(created_blog["_id"]),
        title=created_blog["title"],
        slug=created_blog["slug"],
        content=created_blog["content"],
        excerpt=created_blog["excerpt"],
        coverImage=created_blog.get("coverImage"),
        author=created_blog["author"],
        tags=created_blog["tags"],
        createdAt=created_blog["createdAt"],
        updatedAt=created_blog["updatedAt"]
    )


@app.put("/blogs/{blog_id}", response_model=Blog)
async def update_blog(
        blog_id: str,
        blog_update: BlogUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        blog = db.blogs.find_one({"_id": ObjectId(blog_id)})
        if not blog:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Blog not found",
            )

        update_data = blog_update.dict(exclude_unset=True)
        if update_data:
            if "title" in update_data and not update_data["title"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Title cannot be empty"
                )
            if "slug" in update_data:
                if not update_data["slug"].strip():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Slug cannot be empty"
                    )
                existing_blog = db.blogs.find_one({"slug": update_data["slug"], "_id": {"$ne": ObjectId(blog_id)}})
                if existing_blog:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Blog with this slug already exists",
                    )
            if "content" in update_data and not update_data["content"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Content cannot be empty"
                )
            if "excerpt" in update_data and not update_data["excerpt"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Excerpt cannot be empty"
                )
            if "author" in update_data and not update_data["author"].get("name"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Author name is required"
                )

            update_data["updatedAt"] = datetime.utcnow()
            result = db.blogs.update_one(
                {"_id": ObjectId(blog_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Blog update failed",
                )

        updated_blog = db.blogs.find_one({"_id": ObjectId(blog_id)})

        return Blog(
            id=str(updated_blog["_id"]),
            title=updated_blog["title"],
            slug=updated_blog["slug"],
            content=updated_blog["content"],
            excerpt=updated_blog["excerpt"],
            coverImage=updated_blog.get("coverImage"),
            author=updated_blog["author"],
            tags=updated_blog["tags"],
            createdAt=updated_blog["createdAt"],
            updatedAt=updated_blog["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error updating blog: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Blog update failed",
        )

@app.delete("/blogs/{blog_id}")
async def delete_blog(
        blog_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        # Check if blog exists
        blog = db.blogs.find_one({"_id": ObjectId(blog_id)})
        if not blog:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Blog not found",
            )

        # Delete blog
        result = db.blogs.delete_one({"_id": ObjectId(blog_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Blog deletion failed",
            )

        return {"message": "Blog deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting blog: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Blog deletion failed",
        )

@app.post("/blogs/image")
async def upload_blog_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_admin_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    # Upload to Cloudinary
    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/blogs")

    return {"imageUrl": image_url}


# Emergency Alert routes
@app.get("/emergency-alerts", response_model=List[EmergencyAlert])
async def get_emergency_alerts(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        current_user: UserInDB = Depends(get_current_active_user)
):
    query = {}

    # Regular users can only see their own alerts
    if current_user.role != "admin":
        query["userId"] = str(current_user.id)

    if status:
        query["status"] = status

    alerts = list(db.emergency_alerts.find(query).skip(skip).limit(limit))
    return [
        EmergencyAlert(
            id=str(alert["_id"]),
            userId=alert["userId"],
            message=alert["message"],
            location=alert.get("location"),
            status=alert["status"],
            createdAt=alert["createdAt"],
            updatedAt=alert["updatedAt"]
        ) for alert in alerts
    ]


@app.post("/emergency-alerts", response_model=EmergencyAlert)
async def create_emergency_alert(
        alert: EmergencyAlertCreate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    if not alert.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message is required"
        )

    if alert.userId != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    alert_in_db = EmergencyAlertInDB(**alert.dict())
    result = db.emergency_alerts.insert_one(alert_in_db.dict(by_alias=True))

    created_alert = db.emergency_alerts.find_one({"_id": result.inserted_id})

    user = get_user_by_id(alert.userId)
    if user:
        alert_html = f"""
        <html>
            <body>
                <h1>Emergency Alert Received</h1>
                <p>Dear {user.firstName},</p>
                <p>We have received your emergency alert: {alert.message}</p>
                <p>Our team is responding and will contact you shortly.</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """
        send_email(user.email, "Emergency Alert - Naija Concierge", alert_html)

    notification_message = f"""
    New emergency alert:
    - Client: {user.firstName} {user.lastName}
    - Message: {alert.message}
    - Location: {alert.location or "Not provided"}
    - Status: {created_alert["status"]}
    """
    send_admin_notification("New Emergency Alert", notification_message)

    return EmergencyAlert(
        id=str(created_alert["_id"]),
        userId=created_alert["userId"],
        message=created_alert["message"],
        location=created_alert.get("location"),
        status=created_alert["status"],
        createdAt=created_alert["createdAt"],
        updatedAt=created_alert["updatedAt"]
    )



@app.put("/emergency-alerts/{alert_id}", response_model=EmergencyAlert)
async def update_emergency_alert(
        alert_id: str,
        alert_update: EmergencyAlertUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        alert = db.emergency_alerts.find_one({"_id": ObjectId(alert_id)})
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Emergency alert not found",
            )

        update_data = alert_update.dict(exclude_unset=True)
        if update_data:
            if "message" in update_data and not update_data["message"].strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Message cannot be empty"
                )

            update_data["updatedAt"] = datetime.utcnow()
            result = db.emergency_alerts.update_one(
                {"_id": ObjectId(alert_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Emergency alert update failed",
                )

        updated_alert = db.emergency_alerts.find_one({"_id": ObjectId(alert_id)})

        if "status" in update_data:
            user = get_user_by_id(updated_alert["userId"])
            if user:
                status_update_html = f"""
                <html>
                    <body>
                        <h1>Emergency Alert Status Update</h1>
                        <p>Dear {user.firstName},</p>
                        <p>Your emergency alert has been updated to {updated_alert["status"]}.</p>
                        <p>Message: {updated_alert["message"]}</p>
                        <p>If you need further assistance, please contact us.</p>
                        <p>Best regards,<br>The Naija Concierge Team</p>
                    </body>
                </html>
                """
                send_email(user.email, "Emergency Alert Status Update - Naija Concierge", status_update_html)

        return EmergencyAlert(
            id=str(updated_alert["_id"]),
            userId=updated_alert["userId"],
            message=updated_alert["message"],
            location=updated_alert.get("location"),
            status=updated_alert["status"],
            createdAt=updated_alert["createdAt"],
            updatedAt=updated_alert["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error updating emergency alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Emergency alert update failed",
        )



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



# Admin dashboard stats
@app.get("/admin/stats")
async def get_admin_stats(current_user: UserInDB = Depends(get_admin_user)):
    try:
        # Get total users
        total_users = db.users.count_documents({})

        # Get total bookings
        total_bookings = db.bookings.count_documents({})

        # Get total booking revenue (from successful payments)
        total_booking_revenue = 0
        successful_bookings = list(db.bookings.find({
            "payment_status": "successful",
            "payment_amount": {"$exists": True, "$ne": None}
        }))

        for booking in successful_bookings:
            if booking.get("payment_amount"):
                total_booking_revenue += booking["payment_amount"]

        # Get total service tiers
        total_service_tiers = db.service_tiers.count_documents({})

        # Get user growth (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        new_users = db.users.count_documents({"createdAt": {"$gte": thirty_days_ago}})
        user_growth = (new_users / total_users) * 100 if total_users > 0 else 0

        # Get booking growth (last 30 days)
        new_bookings = db.bookings.count_documents({"createdAt": {"$gte": thirty_days_ago}})
        booking_growth = (new_bookings / total_bookings) * 100 if total_bookings > 0 else 0

        # Get revenue growth (compare last 30 days with previous 30 days)
        last_30_days_bookings = list(db.bookings.find({
            "createdAt": {"$gte": thirty_days_ago},
            "payment_status": "successful",
            "payment_amount": {"$exists": True, "$ne": None}
        }))

        last_30_days_revenue = 0
        for booking in last_30_days_bookings:
            if booking.get("payment_amount"):
                last_30_days_revenue += booking["payment_amount"]

        previous_30_days_start = thirty_days_ago - timedelta(days=30)
        previous_30_days_bookings = list(db.bookings.find({
            "createdAt": {"$gte": previous_30_days_start, "$lt": thirty_days_ago},
            "payment_status": "successful",
            "payment_amount": {"$exists": True, "$ne": None}
        }))

        previous_30_days_revenue = 0
        for booking in previous_30_days_bookings:
            if booking.get("payment_amount"):
                previous_30_days_revenue += booking["payment_amount"]

        revenue_growth = ((
                                      last_30_days_revenue - previous_30_days_revenue) / previous_30_days_revenue) * 100 if previous_30_days_revenue > 0 else 0

        # Get tier growth (compare current tiers with previous month)
        current_tiers = db.service_tiers.count_documents({
            "created_at": {"$gte": thirty_days_ago}
        })
        previous_tiers = db.service_tiers.count_documents({
            "created_at": {"$gte": previous_30_days_start, "$lt": thirty_days_ago}
        })
        tier_growth = ((current_tiers - previous_tiers) / previous_tiers) * 100 if previous_tiers > 0 else 0

        return {
            "totalUsers": total_users,
            "totalBookings": total_bookings,
            "totalBookingRevenue": total_booking_revenue,  # Changed from totalRevenue
            "totalServiceTiers": total_service_tiers,  # Changed from activePackages
            "userGrowth": round(user_growth, 1),
            "bookingGrowth": round(booking_growth, 1),
            "revenueGrowth": round(revenue_growth, 1),
            "tierGrowth": round(tier_growth, 1)  # Changed from packageGrowth
        }
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get admin stats",
        )


@app.get("/analytics/chart-data", response_model=ChartDataResponse)
async def get_chart_data(timeframe: Timeframe = Timeframe.weekly):
    try:
        if timeframe == Timeframe.weekly:
            # Get data for the last 7 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)

            # Generate day names for the past 7 days
            days = [(start_date + timedelta(days=i)).strftime("%a") for i in range(7)]

            booking_data = []
            revenue_data = []

            for i, day in enumerate(days):
                day_start = start_date + timedelta(days=i)
                day_end = day_start + timedelta(days=1)

                # Count bookings for the day
                total_bookings = db.bookings.count_documents({
                    "bookingDate": {"$gte": day_start, "$lt": day_end}
                })

                # Count completed bookings for the day
                completed_bookings = db.bookings.count_documents({
                    "bookingDate": {"$gte": day_start, "$lt": day_end},
                    "status": "completed"
                })

                # Calculate revenue for the day
                day_bookings = db.bookings.find({
                    "bookingDate": {"$gte": day_start, "$lt": day_end},
                    "status": {"$in": ["confirmed", "completed"]}
                })

                day_revenue = 0
                for booking in day_bookings:
                    service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                    if service:
                        day_revenue += service["price"]

                booking_data.append({
                    "name": day,
                    "bookings": total_bookings,
                    "completed": completed_bookings
                })

                revenue_data.append({
                    "name": day,
                    "revenue": day_revenue
                })

        else:  # Monthly timeframe
            # Get data for the last 4 weeks
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(weeks=4)

            booking_data = []
            revenue_data = []

            for week in range(4):
                week_start = start_date + timedelta(weeks=week)
                week_end = week_start + timedelta(weeks=1)

                # Count bookings for the week
                total_bookings = db.bookings.count_documents({
                    "bookingDate": {"$gte": week_start, "$lt": week_end}
                })

                # Count completed bookings for the week
                completed_bookings = db.bookings.count_documents({
                    "bookingDate": {"$gte": week_start, "$lt": week_end},
                    "status": "completed"
                })

                # Calculate revenue for the week
                week_bookings = db.bookings.find({
                    "bookingDate": {"$gte": week_start, "$lt": week_end},
                    "status": {"$in": ["confirmed", "completed"]}
                })

                week_revenue = 0
                for booking in week_bookings:
                    service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                    if service:
                        week_revenue += service["price"]

                booking_data.append({
                    "name": f"Week {week + 1}",
                    "bookings": total_bookings,
                    "completed": completed_bookings
                })

                revenue_data.append({
                    "name": f"Week {week + 1}",
                    "revenue": week_revenue
                })

        return {
            "bookingData": booking_data,
            "revenueData": revenue_data
        }

    except Exception as e:
        logger.error(f"Error fetching chart data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chart data"
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

@app.post("/newsletter/unsubscribe")
async def unsubscribe_from_newsletter(
        email: EmailStr = Form(...)
):
    """
    Unsubscribe from the newsletter
    """
    try:
        result = db.newsletter_subscribers.update_one(
            {"email": email, "is_active": True},
            {"$set": {"is_active": False}}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Email not found in active subscriptions"
            )

        # Send confirmation email
        confirmation_html = f"""
        <html>
            <body>
                <h1>You're Unsubscribed</h1>
                <p>You've been successfully unsubscribed from our newsletter.</p>
                <p>We're sorry to see you go. You can resubscribe anytime.</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """

        send_email(
            email,
            "You're Unsubscribed - Naija Concierge",
            confirmation_html
        )

        return {"message": "Successfully unsubscribed"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unsubscribing from newsletter: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process unsubscribe request"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
