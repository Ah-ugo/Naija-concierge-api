from fastapi import FastAPI, Depends, HTTPException, status, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, GetJsonSchemaHandler, BeforeValidator
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

# Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")

smtp_server = "smtp.gmail.com"
smtp_port = 465
email_address = "ahuekweprinceugo@gmail.com"  # Make sure this matches your sending address
password = os.getenv("GMAIL_PASS")  # Use the environment variable
recipient_email = "ahuekweprinceugo@gmail.com" #change to a test email

print(password)
subject = "Test Email from Python"  # Added a subject
message_text = "This is a test email sent using smtplib and a context manager."  # Added message

msg = MIMEText(message_text)
msg['Subject'] = subject
msg['From'] = email_address
msg['To'] = recipient_email

try:
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.ehlo()  # Identify ourselves to the server
        # server.starttls()  # Put the connection in TLS mode
        # server.ehlo()  # Re-identify after TLS negotiation
        server.login(email_address, password)
        server.sendmail(email_address, [recipient_email], msg.as_string())
    print("Email sent successfully!")
except Exception as e:
    print(f"Error sending email: {e}")


# Pydantic models
# class PyObjectId(ObjectId):
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate
#
#     @classmethod
#     def validate(cls, v, handler):
#         if not ObjectId.is_valid(v):
#             raise ValueError("Invalid ObjectId")
#         return ObjectId(v)
#
#     @classmethod
#     def __get_pydantic_core_schema__(
#         cls,
#         source: type[Any],
#         handler: GetJsonSchemaHandler,
#     ) -> core_schema.CoreSchema:
#         return core_schema.str_schema()

PyObjectId = Annotated[str, BeforeValidator(str)]

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
    # id: Optional[Annotated[PyObjectId, Field(alias="_id", default=None)]]
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

    # Model validators for Pydantic v2
    @classmethod
    def model_validate(cls, obj, **kwargs):
        # Convert ObjectId to string if present
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


class ServiceBase(BaseModel):
    name: str
    description: str
    category: str
    price: float
    image: Optional[str] = None
    duration: str
    isAvailable: bool = True


class ServiceCreate(ServiceBase):
    pass


class ServiceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    image: Optional[str] = None
    duration: Optional[str] = None
    isAvailable: Optional[bool] = None


class ServiceInDB(ServiceBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
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

    class Config:
        orm_mode = True


class PackageBase(BaseModel):
    name: str
    description: str
    price: float
    duration: str
    features: List[str]
    image: Optional[str] = None
    type: str
    isPopular: bool = False


class PackageCreate(PackageBase):
    pass


class PackageUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    duration: Optional[str] = None
    features: Optional[List[str]] = None
    image: Optional[str] = None
    type: Optional[str] = None
    isPopular: Optional[bool] = None


class PackageInDB(PackageBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Package(PackageBase):
    id: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True


class BookingBase(BaseModel):
    userId: str
    serviceId: str
    bookingDate: datetime
    status: str = "pending"
    specialRequests: Optional[str] = None


class BookingCreate(BookingBase):
    pass


class BookingUpdate(BaseModel):
    bookingDate: Optional[datetime] = None
    status: Optional[str] = None
    specialRequests: Optional[str] = None


class BookingInDB(BookingBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Booking(BookingBase):
    id: str
    createdAt: datetime
    updatedAt: datetime
    service: Optional[Service] = None

    class Config:
        orm_mode = True


class SubscriptionBase(BaseModel):
    userId: str
    packageId: str
    startDate: datetime
    endDate: datetime
    status: str = "active"


class SubscriptionCreate(SubscriptionBase):
    pass


class SubscriptionUpdate(BaseModel):
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    status: Optional[str] = None


class SubscriptionInDB(SubscriptionBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Subscription(SubscriptionBase):
    id: str
    createdAt: datetime
    updatedAt: datetime
    package: Optional[Package] = None

    class Config:
        orm_mode = True


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


class DocumentBase(BaseModel):
    userId: str
    name: str
    type: str
    url: str


class DocumentCreate(DocumentBase):
    pass


class DocumentInDB(DocumentBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    uploadDate: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Document(DocumentBase):
    id: str
    uploadDate: datetime

    class Config:
        orm_mode = True


# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(email: str):
    user = db.users.find_one({"email": email})
    if user:
        user["_id"] = str(user["_id"])  # Convert ObjectId to string
        return UserInDB(**user)
    return None


def get_user_by_id(user_id: str):
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if user:
            return UserInDB(**user)
    except Exception as e:
        logger.error(f"Error getting user by ID: {e}")
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


def send_email(to_email: str, subject: str, html_content: str):
    """Send email using SMTP, with improved connection handling."""
    smtp_server = os.getenv("SMTP_HOST")  # Use SMTP_HOST
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("GMAIL_ADDRESS")  # Use GMAIL_ADDRESS, ensure it's the sender

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = to_email

    html_part = MIMEText(html_content, "html")
    message.attach(html_part)

    server = None  # Initialize server outside the try block
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        # server.starttls()  # Upgrade to secure connection
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, to_email, message.as_string())
        logger.info(f"Email sent successfully to {to_email}")  # Add logging
        return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False
    finally:
        if server:
            try:
                server.quit()  # Ensure server.quit() is always called
            except Exception as e:
                logger.error(f"Error closing SMTP connection: {e}")

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


def serialize_object_id(obj):
    """Convert ObjectId to string in a dictionary"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, ObjectId):
                obj[k] = str(v)
            elif isinstance(v, dict):
                serialize_object_id(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        serialize_object_id(item)
    return obj


# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Naija Concierge API"}


# Auth routes
@app.post("/auth/register", response_model=Token)
async def register(user: UserCreate):
    # Check if user already exists
    if db.users.find_one({"email": user.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    del user_dict["password"]
    user_in_db = UserInDB(
        **user_dict,
        hashed_password=hashed_password,
        role="user"
    )

    # Insert into database
    result = db.users.insert_one(user_in_db.dict(by_alias=True))

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    # Get created user
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

    # Send welcome email
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


# User routes
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


@app.post("/users/profile-image")
async def upload_profile_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_current_user)
):
    # if not file.content_type.startswith("image/"):
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="File must be an image"
    #     )
    print(current_user)
    # Upload to Cloudinary
    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/profiles")

    # Update user profile
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


# Service routes
@app.get("/services", response_model=List[Service])
async def get_services(
        skip: int = 0,
        limit: int = 100,
        category: Optional[str] = None
):
    query = {}
    if category:
        query["category"] = category

    services = list(db.services.find(query).skip(skip).limit(limit))
    return [
        Service(
            id=str(service["_id"]),
            name=service["name"],
            description=service["description"],
            category=service["category"],
            price=service["price"],
            image=service.get("image"),
            duration=service["duration"],
            isAvailable=service["isAvailable"],
            createdAt=service["createdAt"],
            updatedAt=service["updatedAt"]
        ) for service in services
    ]


@app.get("/services/{service_id}", response_model=Service)
async def get_service(service_id: str):
    try:
        service = db.services.find_one({"_id": ObjectId(service_id)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found",
            )

        return Service(
            id=str(service["_id"]),
            name=service["name"],
            description=service["description"],
            category=service["category"],
            price=service["price"],
            image=service.get("image"),
            duration=service["duration"],
            isAvailable=service["isAvailable"],
            createdAt=service["createdAt"],
            updatedAt=service["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error getting service: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found",
        )


@app.post("/services", response_model=Service)
async def create_service(
        service: ServiceCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    service_in_db = ServiceInDB(**service.dict())
    result = db.services.insert_one(service_in_db.dict(by_alias=True))

    created_service = db.services.find_one({"_id": result.inserted_id})

    return Service(
        id=str(created_service["_id"]),
        name=created_service["name"],
        description=created_service["description"],
        category=created_service["category"],
        price=created_service["price"],
        image=created_service.get("image"),
        duration=created_service["duration"],
        isAvailable=created_service["isAvailable"],
        createdAt=created_service["createdAt"],
        updatedAt=created_service["updatedAt"]
    )


@app.put("/services/{service_id}", response_model=Service)
async def update_service(
        service_id: str,
        service_update: ServiceUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        # Check if service exists
        service = db.services.find_one({"_id": ObjectId(service_id)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found",
            )

        # Update service
        update_data = service_update.dict(exclude_unset=True)
        if update_data:
            update_data["updatedAt"] = datetime.utcnow()
            result = db.services.update_one(
                {"_id": ObjectId(service_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Service update failed",
                )

        # Get updated service
        updated_service = db.services.find_one({"_id": ObjectId(service_id)})

        return Service(
            id=str(updated_service["_id"]),
            name=updated_service["name"],
            description=updated_service["description"],
            category=updated_service["category"],
            price=updated_service["price"],
            image=updated_service.get("image"),
            duration=updated_service["duration"],
            isAvailable=updated_service["isAvailable"],
            createdAt=updated_service["createdAt"],
            updatedAt=updated_service["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error updating service: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Service update failed",
        )


@app.delete("/services/{service_id}")
async def delete_service(
        service_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        # Check if service exists
        service = db.services.find_one({"_id": ObjectId(service_id)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found",
            )

        # Check if service has bookings
        bookings = db.bookings.find_one({"serviceId": service_id})
        if bookings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete service with existing bookings",
            )

        # Delete service
        result = db.services.delete_one({"_id": ObjectId(service_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Service deletion failed",
            )

        return {"message": "Service deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting service: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Service deletion failed",
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

    # Upload to Cloudinary
    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/services")

    return {"imageUrl": image_url}


# Package routes
@app.get("/packages", response_model=List[Package])
async def get_packages(
        skip: int = 0,
        limit: int = 100,
        type: Optional[str] = None
):
    query = {}
    if type:
        query["type"] = type

    packages = list(db.packages.find(query).skip(skip).limit(limit))
    return [
        Package(
            id=str(package["_id"]),
            name=package["name"],
            description=package["description"],
            price=package["price"],
            duration=package["duration"],
            features=package["features"],
            image=package.get("image"),
            type=package["type"],
            isPopular=package["isPopular"],
            createdAt=package["createdAt"],
            updatedAt=package["updatedAt"]
        ) for package in packages
    ]


@app.get("/packages/{package_id}", response_model=Package)
async def get_package(package_id: str):
    try:
        package = db.packages.find_one({"_id": ObjectId(package_id)})
        if not package:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Package not found",
            )

        return Package(
            id=str(package["_id"]),
            name=package["name"],
            description=package["description"],
            price=package["price"],
            duration=package["duration"],
            features=package["features"],
            image=package.get("image"),
            type=package["type"],
            isPopular=package["isPopular"],
            createdAt=package["createdAt"],
            updatedAt=package["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error getting package: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Package not found",
        )


@app.post("/packages", response_model=Package)
async def create_package(
        package: PackageCreate,
        current_user: UserInDB = Depends(get_admin_user)
):
    package_in_db = PackageInDB(**package.dict())
    result = db.packages.insert_one(package_in_db.dict(by_alias=True))

    created_package = db.packages.find_one({"_id": result.inserted_id})

    return Package(
        id=str(created_package["_id"]),
        name=created_package["name"],
        description=created_package["description"],
        price=created_package["price"],
        duration=created_package["duration"],
        features=created_package["features"],
        image=created_package.get("image"),
        type=created_package["type"],
        isPopular=created_package["isPopular"],
        createdAt=created_package["createdAt"],
        updatedAt=created_package["updatedAt"]
    )


@app.put("/packages/{package_id}", response_model=Package)
async def update_package(
        package_id: str,
        package_update: PackageUpdate,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        # Check if package exists
        package = db.packages.find_one({"_id": ObjectId(package_id)})
        if not package:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Package not found",
            )

        # Update package
        update_data = package_update.dict(exclude_unset=True)
        if update_data:
            update_data["updatedAt"] = datetime.utcnow()
            result = db.packages.update_one(
                {"_id": ObjectId(package_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Package update failed",
                )

        # Get updated package
        updated_package = db.packages.find_one({"_id": ObjectId(package_id)})

        return Package(
            id=str(updated_package["_id"]),
            name=updated_package["name"],
            description=updated_package["description"],
            price=updated_package["price"],
            duration=updated_package["duration"],
            features=updated_package["features"],
            image=updated_package.get("image"),
            type=updated_package["type"],
            isPopular=updated_package["isPopular"],
            createdAt=updated_package["createdAt"],
            updatedAt=updated_package["updatedAt"]
        )
    except Exception as e:
        logger.error(f"Error updating package: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Package update failed",
        )


@app.delete("/packages/{package_id}")
async def delete_package(
        package_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        # Check if package exists
        package = db.packages.find_one({"_id": ObjectId(package_id)})
        if not package:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Package not found",
            )

        # Check if package has subscriptions
        subscriptions = db.subscriptions.find_one({"packageId": package_id})
        if subscriptions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete package with existing subscriptions",
            )

        # Delete package
        result = db.packages.delete_one({"_id": ObjectId(package_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Package deletion failed",
            )

        return {"message": "Package deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting package: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Package deletion failed",
        )


@app.post("/packages/image")
async def upload_package_image(
        file: UploadFile = File(...),
        current_user: UserInDB = Depends(get_admin_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    # Upload to Cloudinary
    image_url = upload_file_to_cloudinary(file, folder="naija_concierge/packages")

    return {"imageUrl": image_url}


# Booking routes
@app.get("/bookings", response_model=List[Booking])
async def get_bookings(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        current_user: UserInDB = Depends(get_current_active_user)
):
    query = {}

    # Regular users can only see their own bookings
    if current_user.role != "admin":
        query["userId"] = str(current_user.id)

    if status:
        query["status"] = status

    bookings = list(db.bookings.find(query).skip(skip).limit(limit))
    result = []

    for booking in bookings:
        try:
            service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
            service_obj = None
            if service:
                service_obj = Service(
                    id=str(service["_id"]),
                    name=service["name"],
                    description=service["description"],
                    category=service["category"],
                    price=service["price"],
                    image=service.get("image"),
                    duration=service["duration"],
                    isAvailable=service["isAvailable"],
                    createdAt=service["createdAt"],
                    updatedAt=service["updatedAt"]
                )

            result.append(
                Booking(
                    id=str(booking["_id"]),
                    userId=booking["userId"],
                    serviceId=booking["serviceId"],
                    bookingDate=booking["bookingDate"],
                    status=booking["status"],
                    specialRequests=booking.get("specialRequests"),
                    createdAt=booking["createdAt"],
                    updatedAt=booking["updatedAt"],
                    service=service_obj
                )
            )
        except Exception as e:
            logger.error(f"Error processing booking: {e}")

    return result


@app.get("/bookings/{booking_id}", response_model=Booking)
async def get_booking(
        booking_id: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found",
            )

        # Regular users can only see their own bookings
        if current_user.role != "admin" and booking["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
        service_obj = None
        if service:
            service_obj = Service(
                id=str(service["_id"]),
                name=service["name"],
                description=service["description"],
                category=service["category"],
                price=service["price"],
                image=service.get("image"),
                duration=service["duration"],
                isAvailable=service["isAvailable"],
                createdAt=service["createdAt"],
                updatedAt=service["updatedAt"]
            )

        return Booking(
            id=str(booking["_id"]),
            userId=booking["userId"],
            serviceId=booking["serviceId"],
            bookingDate=booking["bookingDate"],
            status=booking["status"],
            specialRequests=booking.get("specialRequests"),
            createdAt=booking["createdAt"],
            updatedAt=booking["updatedAt"],
            service=service_obj
        )
    except Exception as e:
        logger.error(f"Error getting booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Booking not found",
        )


@app.post("/bookings", response_model=Booking)
async def create_booking(
        booking: BookingCreate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    # Regular users can only create bookings for themselves
    if current_user.role != "admin" and booking.userId != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    # Check if service exists
    try:
        service = db.services.find_one({"_id": ObjectId(booking.serviceId)})
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found",
            )

        # Check if service is available
        if not service["isAvailable"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Service is not available",
            )
    except Exception as e:
        logger.error(f"Error checking service: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found",
        )

    booking_in_db = BookingInDB(**booking.dict())
    result = db.bookings.insert_one(booking_in_db.dict(by_alias=True))

    created_booking = db.bookings.find_one({"_id": result.inserted_id})

    service_obj = Service(
        id=str(service["_id"]),
        name=service["name"],
        description=service["description"],
        category=service["category"],
        price=service["price"],
        image=service.get("image"),
        duration=service["duration"],
        isAvailable=service["isAvailable"],
        createdAt=service["createdAt"],
        updatedAt=service["updatedAt"]
    )

    # Send booking confirmation email
    user = get_user_by_id(booking.userId)
    if user:
        booking_html = f"""
        <html>
            <body>
                <h1>Booking Confirmation</h1>
                <p>Dear {user.firstName},</p>
                <p>Your booking for {service["name"]} has been received and is currently {created_booking["status"]}.</p>
                <p>Booking Details:</p>
                <ul>
                    <li>Service: {service["name"]}</li>
                    <li>Date: {created_booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}</li>
                    <li>Status: {created_booking["status"]}</li>
                    <li>Price: ₦{service["price"]}</li>
                </ul>
                <p>We will contact you shortly to confirm your booking.</p>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """
        send_email(user.email, "Booking Confirmation - Naija Concierge", booking_html)

    return Booking(
        id=str(created_booking["_id"]),
        userId=created_booking["userId"],
        serviceId=created_booking["serviceId"],
        bookingDate=created_booking["bookingDate"],
        status=created_booking["status"],
        specialRequests=created_booking.get("specialRequests"),
        createdAt=created_booking["createdAt"],
        updatedAt=created_booking["updatedAt"],
        service=service_obj
    )


@app.put("/bookings/{booking_id}", response_model=Booking)
async def update_booking(
        booking_id: str,
        booking_update: BookingUpdate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        # Check if booking exists
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found",
            )

        # Regular users can only update their own bookings
        if current_user.role != "admin" and booking["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        # Regular users can only update certain fields
        update_data = booking_update.dict(exclude_unset=True)
        if current_user.role != "admin" and "status" in update_data:
            # Regular users can only cancel their bookings
            if update_data["status"] != "cancelled":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions to change status",
                )

        if update_data:
            update_data["updatedAt"] = datetime.utcnow()
            result = db.bookings.update_one(
                {"_id": ObjectId(booking_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Booking update failed",
                )

        # Get updated booking
        updated_booking = db.bookings.find_one({"_id": ObjectId(booking_id)})

        service = db.services.find_one({"_id": ObjectId(updated_booking["serviceId"])})
        service_obj = None
        if service:
            service_obj = Service(
                id=str(service["_id"]),
                name=service["name"],
                description=service["description"],
                category=service["category"],
                price=service["price"],
                image=service.get("image"),
                duration=service["duration"],
                isAvailable=service["isAvailable"],
                createdAt=service["createdAt"],
                updatedAt=service["updatedAt"]
            )

        # Send booking update email if status changed
        if "status" in update_data:
            user = get_user_by_id(updated_booking["userId"])
            if user:
                status_update_html = f"""
                <html>
                    <body>
                        <h1>Booking Status Update</h1>
                        <p>Dear {user.firstName},</p>
                        <p>Your booking for {service["name"]} has been updated to {updated_booking["status"]}.</p>
                        <p>Booking Details:</p>
                        <ul>
                            <li>Service: {service["name"]}</li>
                            <li>Date: {updated_booking["bookingDate"].strftime("%Y-%m-%d %H:%M")}</li>
                            <li>Status: {updated_booking["status"]}</li>
                        </ul>
                        <p>If you have any questions, please contact us.</p>
                        <p>Best regards,<br>The Naija Concierge Team</p>
                    </body>
                </html>
                """
                send_email(user.email, "Booking Status Update - Naija Concierge", status_update_html)

        return Booking(
            id=str(updated_booking["_id"]),
            userId=updated_booking["userId"],
            serviceId=updated_booking["serviceId"],
            bookingDate=updated_booking["bookingDate"],
            status=updated_booking["status"],
            specialRequests=updated_booking.get("specialRequests"),
            createdAt=updated_booking["createdAt"],
            updatedAt=updated_booking["updatedAt"],
            service=service_obj
        )
    except Exception as e:
        logger.error(f"Error updating booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Booking update failed",
        )


@app.delete("/bookings/{booking_id}")
async def delete_booking(
        booking_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        # Check if booking exists
        booking = db.bookings.find_one({"_id": ObjectId(booking_id)})
        if not booking:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Booking not found",
            )

        # Delete booking
        result = db.bookings.delete_one({"_id": ObjectId(booking_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Booking deletion failed",
            )

        return {"message": "Booking deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting booking: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Booking deletion failed",
        )


# Subscription routes
@app.get("/subscriptions", response_model=List[Subscription])
async def get_subscriptions(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        current_user: UserInDB = Depends(get_current_active_user)
):
    query = {}

    # Regular users can only see their own subscriptions
    if current_user.role != "admin":
        query["userId"] = str(current_user.id)

    if status:
        query["status"] = status

    subscriptions = list(db.subscriptions.find(query).skip(skip).limit(limit))
    result = []

    for subscription in subscriptions:
        try:
            package = db.packages.find_one({"_id": ObjectId(subscription["packageId"])})
            package_obj = None
            if package:
                package_obj = Package(
                    id=str(package["_id"]),
                    name=package["name"],
                    description=package["description"],
                    price=package["price"],
                    duration=package["duration"],
                    features=package["features"],
                    image=package.get("image"),
                    type=package["type"],
                    isPopular=package["isPopular"],
                    createdAt=package["createdAt"],
                    updatedAt=package["updatedAt"]
                )

            result.append(
                Subscription(
                    id=str(subscription["_id"]),
                    userId=subscription["userId"],
                    packageId=subscription["packageId"],
                    startDate=subscription["startDate"],
                    endDate=subscription["endDate"],
                    status=subscription["status"],
                    createdAt=subscription["createdAt"],
                    updatedAt=subscription["updatedAt"],
                    package=package_obj
                )
            )
        except Exception as e:
            logger.error(f"Error processing subscription: {e}")

    return result


@app.get("/subscriptions/{subscription_id}", response_model=Subscription)
async def get_subscription(
        subscription_id: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        subscription = db.subscriptions.find_one({"_id": ObjectId(subscription_id)})
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found",
            )

        # Regular users can only see their own subscriptions
        if current_user.role != "admin" and subscription["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        package = db.packages.find_one({"_id": ObjectId(subscription["packageId"])})
        package_obj = None
        if package:
            package_obj = Package(
                id=str(package["_id"]),
                name=package["name"],
                description=package["description"],
                price=package["price"],
                duration=package["duration"],
                features=package["features"],
                image=package.get("image"),
                type=package["type"],
                isPopular=package["isPopular"],
                createdAt=package["createdAt"],
                updatedAt=package["updatedAt"]
            )

        return Subscription(
            id=str(subscription["_id"]),
            userId=subscription["userId"],
            packageId=subscription["packageId"],
            startDate=subscription["startDate"],
            endDate=subscription["endDate"],
            status=subscription["status"],
            createdAt=subscription["createdAt"],
            updatedAt=subscription["updatedAt"],
            package=package_obj
        )
    except Exception as e:
        logger.error(f"Error getting subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found",
        )


@app.post("/subscriptions", response_model=Subscription)
async def create_subscription(
        subscription: SubscriptionCreate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    # Regular users can only create subscriptions for themselves
    if current_user.role != "admin" and subscription.userId != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    # Check if package exists
    try:
        package = db.packages.find_one({"_id": ObjectId(subscription.packageId)})
        if not package:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Package not found",
            )
    except Exception as e:
        logger.error(f"Error checking package: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Package not found",
        )

    # Check if user already has an active subscription
    active_subscription = db.subscriptions.find_one({
        "userId": subscription.userId,
        "status": "active"
    })
    if active_subscription:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already has an active subscription",
        )

    subscription_in_db = SubscriptionInDB(**subscription.dict())
    result = db.subscriptions.insert_one(subscription_in_db.dict(by_alias=True))

    created_subscription = db.subscriptions.find_one({"_id": result.inserted_id})

    package_obj = Package(
        id=str(package["_id"]),
        name=package["name"],
        description=package["description"],
        price=package["price"],
        duration=package["duration"],
        features=package["features"],
        image=package.get("image"),
        type=package["type"],
        isPopular=package["isPopular"],
        createdAt=package["createdAt"],
        updatedAt=package["updatedAt"]
    )

    # Send subscription confirmation email
    user = get_user_by_id(subscription.userId)
    if user:
        subscription_html = f"""
        <html>
            <body>
                <h1>Subscription Confirmation</h1>
                <p>Dear {user.firstName},</p>
                <p>Thank you for subscribing to our {package["name"]} package.</p>
                <p>Subscription Details:</p>
                <ul>
                    <li>Package: {package["name"]}</li>
                    <li>Start Date: {created_subscription["startDate"].strftime("%Y-%m-%d")}</li>
                    <li>End Date: {created_subscription["endDate"].strftime("%Y-%m-%d")}</li>
                    <li>Price: ₦{package["price"]}</li>
                </ul>
                <p>Included Services:</p>
                <ul>
                    {"".join([f"<li>{feature}</li>" for feature in package["features"]])}
                </ul>
                <p>Best regards,<br>The Naija Concierge Team</p>
            </body>
        </html>
        """
        send_email(user.email, "Subscription Confirmation - Naija Concierge", subscription_html)

    return Subscription(
        id=str(created_subscription["_id"]),
        userId=created_subscription["userId"],
        packageId=created_subscription["packageId"],
        startDate=created_subscription["startDate"],
        endDate=created_subscription["endDate"],
        status=created_subscription["status"],
        createdAt=created_subscription["createdAt"],
        updatedAt=created_subscription["updatedAt"],
        package=package_obj
    )


@app.put("/subscriptions/{subscription_id}", response_model=Subscription)
async def update_subscription(
        subscription_id: str,
        subscription_update: SubscriptionUpdate,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        # Check if subscription exists
        subscription = db.subscriptions.find_one({"_id": ObjectId(subscription_id)})
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found",
            )

        # Regular users can only update their own subscriptions
        if current_user.role != "admin" and subscription["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        # Regular users can only update certain fields
        update_data = subscription_update.dict(exclude_unset=True)
        if current_user.role != "admin" and "status" in update_data:
            # Regular users can only cancel their subscriptions
            if update_data["status"] != "cancelled":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions to change status",
                )

        if update_data:
            update_data["updatedAt"] = datetime.utcnow()
            result = db.subscriptions.update_one(
                {"_id": ObjectId(subscription_id)},
                {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Subscription update failed",
                )

        # Get updated subscription
        updated_subscription = db.subscriptions.find_one({"_id": ObjectId(subscription_id)})

        package = db.packages.find_one({"_id": ObjectId(updated_subscription["packageId"])})
        package_obj = None
        if package:
            package_obj = Package(
                id=str(package["_id"]),
                name=package["name"],
                description=package["description"],
                price=package["price"],
                duration=package["duration"],
                features=package["features"],
                image=package.get("image"),
                type=package["type"],
                isPopular=package["isPopular"],
                createdAt=package["createdAt"],
                updatedAt=package["updatedAt"]
            )

        # Send subscription update email if status changed
        if "status" in update_data:
            user = get_user_by_id(updated_subscription["userId"])
            if user:
                status_update_html = f"""
                <html>
                    <body>
                        <h1>Subscription Status Update</h1>
                        <p>Dear {user.firstName},</p>
                        <p>Your subscription to the {package["name"]} package has been updated to {updated_subscription["status"]}.</p>
                        <p>Subscription Details:</p>
                        <ul>
                            <li>Package: {package["name"]}</li>
                            <li>Start Date: {updated_subscription["startDate"].strftime("%Y-%m-%d")}</li>
                            <li>End Date: {updated_subscription["endDate"].strftime("%Y-%m-%d")}</li>
                            <li>Status: {updated_subscription["status"]}</li>
                        </ul>
                        <p>If you have any questions, please contact us.</p>
                        <p>Best regards,<br>The Naija Concierge Team</p>
                    </body>
                </html>
                """
                send_email(user.email, "Subscription Status Update - Naija Concierge", status_update_html)

        return Subscription(
            id=str(updated_subscription["_id"]),
            userId=updated_subscription["userId"],
            packageId=updated_subscription["packageId"],
            startDate=updated_subscription["startDate"],
            endDate=updated_subscription["endDate"],
            status=updated_subscription["status"],
            createdAt=updated_subscription["createdAt"],
            updatedAt=updated_subscription["updatedAt"],
            package=package_obj
        )
    except Exception as e:
        logger.error(f"Error updating subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subscription update failed",
        )


@app.delete("/subscriptions/{subscription_id}")
async def delete_subscription(
        subscription_id: str,
        current_user: UserInDB = Depends(get_admin_user)
):
    try:
        # Check if subscription exists
        subscription = db.subscriptions.find_one({"_id": ObjectId(subscription_id)})
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found",
            )

        # Delete subscription
        result = db.subscriptions.delete_one({"_id": ObjectId(subscription_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Subscription deletion failed",
            )

        return {"message": "Subscription deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subscription deletion failed",
        )


# Blog routes
@app.get("/blogs", response_model=List[Blog])
async def get_blogs(
        skip: int = 0,
        limit: int = 100,
        tag: Optional[str] = None
):
    query = {}
    if tag:
        query["tags"] = tag

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


@app.get("/blogs/{slug}", response_model=Blog)
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
    # Check if slug already exists
    existing_blog = db.blogs.find_one({"slug": blog.slug})
    if existing_blog:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Slug already exists",
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
        # Check if blog exists
        blog = db.blogs.find_one({"_id": ObjectId(blog_id)})
        if not blog:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Blog not found",
            )

        # Check if slug already exists (if updating slug)
        if blog_update.slug and blog_update.slug != blog["slug"]:
            existing_blog = db.blogs.find_one({"slug": blog_update.slug})
            if existing_blog:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Slug already exists",
                )

        # Update blog
        update_data = blog_update.dict(exclude_unset=True)
        if update_data:
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

        # Get updated blog
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
    # Regular users can only create alerts for themselves
    if current_user.role != "admin" and alert.userId != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    alert_in_db = EmergencyAlertInDB(**alert.dict())
    result = db.emergency_alerts.insert_one(alert_in_db.dict(by_alias=True))

    created_alert = db.emergency_alerts.find_one({"_id": result.inserted_id})

    # Send emergency notification to admin
    admin_users = list(db.users.find({"role": "admin"}))
    for admin in admin_users:
        emergency_html = f"""
        <html>
            <body>
                <h1 style="color: red;">EMERGENCY ALERT</h1>
                <p>An emergency alert has been submitted by a user.</p>
                <p><strong>User ID:</strong> {alert.userId}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Location:</strong> {alert.location or "Not provided"}</p>
                <p><strong>Time:</strong> {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
                <p>Please respond immediately.</p>
            </body>
        </html>
        """
        send_email(admin["email"], "EMERGENCY ALERT - Naija Concierge", emergency_html)

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
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        # Check if alert exists
        alert = db.emergency_alerts.find_one({"_id": ObjectId(alert_id)})
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Emergency alert not found",
            )

        # Regular users can only update their own alerts
        if current_user.role != "admin" and alert["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        # Update alert
        update_data = alert_update.dict(exclude_unset=True)
        if update_data:
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

        # Get updated alert
        updated_alert = db.emergency_alerts.find_one({"_id": ObjectId(alert_id)})

        # Notify user if status changed to resolved
        if "status" in update_data and update_data["status"] == "resolved":
            user = get_user_by_id(updated_alert["userId"])
            if user:
                resolved_html = f"""
                <html>
                    <body>
                        <h1>Emergency Alert Resolved</h1>
                        <p>Dear {user.firstName},</p>
                        <p>Your emergency alert has been resolved.</p>
                        <p><strong>Message:</strong> {updated_alert["message"]}</p>
                        <p><strong>Location:</strong> {updated_alert.get("location") or "Not provided"}</p>
                        <p><strong>Status:</strong> {updated_alert["status"]}</p>
                        <p>If you have any further concerns, please contact us.</p>
                        <p>Best regards,<br>The Naija Concierge Team</p>
                    </body>
                </html>
                """
                send_email(user.email, "Emergency Alert Resolved - Naija Concierge", resolved_html)

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
        # Store message in database
        message_dict = message.dict()
        message_dict["createdAt"] = datetime.utcnow()
        db.contact_messages.insert_one(message_dict)

        # Send notification email to admin
        admin_users = list(db.users.find({"role": "admin"}))
        for admin in admin_users:
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
            send_email(admin["email"], f"New Contact Message: {message.subject}", contact_html)

        # Send confirmation email to user
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
            detail="Failed to send contact message",
        )


# Document routes
@app.get("/documents", response_model=List[Document])
async def get_documents(
        current_user: UserInDB = Depends(get_current_active_user)
):
    query = {}

    # Regular users can only see their own documents
    if current_user.role != "admin":
        query["userId"] = str(current_user.id)

    documents = list(db.documents.find(query))
    return [
        Document(
            id=str(doc["_id"]),
            userId=doc["userId"],
            name=doc["name"],
            type=doc["type"],
            url=doc["url"],
            uploadDate=doc["uploadDate"]
        ) for doc in documents
    ]


@app.post("/documents", response_model=Document)
async def upload_document(
        file: UploadFile = File(...),
        userId: str = Form(...),
        documentType: str = Form(...),
        current_user: UserInDB = Depends(get_current_active_user)
):
    # Regular users can only upload documents for themselves
    if current_user.role != "admin" and userId != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    # Upload to Cloudinary
    document_url = upload_file_to_cloudinary(file, folder=f"naija_concierge/documents/{documentType}")

    # Create document record
    document = DocumentInDB(
        userId=userId,
        name=file.filename,
        type=documentType,
        url=document_url
    )

    result = db.documents.insert_one(document.dict(by_alias=True))
    created_document = db.documents.find_one({"_id": result.inserted_id})

    return Document(
        id=str(created_document["_id"]),
        userId=created_document["userId"],
        name=created_document["name"],
        type=created_document["type"],
        url=created_document["url"],
        uploadDate=created_document["uploadDate"]
    )


@app.delete("/documents/{document_id}")
async def delete_document(
        document_id: str,
        current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        # Check if document exists
        document = db.documents.find_one({"_id": ObjectId(document_id)})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

        # Regular users can only delete their own documents
        if current_user.role != "admin" and document["userId"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )

        # Delete document
        result = db.documents.delete_one({"_id": ObjectId(document_id)})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document deletion failed",
            )

        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document deletion failed",
        )


# Admin dashboard stats
@app.get("/admin/stats")
async def get_admin_stats(current_user: UserInDB = Depends(get_admin_user)):
    try:
        # Get total users
        total_users = db.users.count_documents({})

        # Get total bookings
        total_bookings = db.bookings.count_documents({})

        # Get total revenue
        bookings = list(db.bookings.find({"status": {"$in": ["confirmed", "completed"]}}))
        total_revenue = 0
        for booking in bookings:
            try:
                service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                if service:
                    total_revenue += service["price"]
            except Exception:
                pass

        # Get active packages
        active_packages = db.subscriptions.count_documents({"status": "active"})

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
            "status": {"$in": ["confirmed", "completed"]}
        }))
        last_30_days_revenue = 0
        for booking in last_30_days_bookings:
            try:
                service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                if service:
                    last_30_days_revenue += service["price"]
            except Exception:
                pass

        previous_30_days_start = thirty_days_ago - timedelta(days=30)
        previous_30_days_bookings = list(db.bookings.find({
            "createdAt": {"$gte": previous_30_days_start, "$lt": thirty_days_ago},
            "status": {"$in": ["confirmed", "completed"]}
        }))
        previous_30_days_revenue = 0
        for booking in previous_30_days_bookings:
            try:
                service = db.services.find_one({"_id": ObjectId(booking["serviceId"])})
                if service:
                    previous_30_days_revenue += service["price"]
            except Exception:
                pass

        revenue_growth = ((
                                      last_30_days_revenue - previous_30_days_revenue) / previous_30_days_revenue) * 100 if previous_30_days_revenue > 0 else 0

        # Get package growth (compare active packages with previous month)
        current_active_packages = db.subscriptions.count_documents({
            "status": "active",
            "startDate": {"$gte": thirty_days_ago}
        })
        previous_active_packages = db.subscriptions.count_documents({
            "status": "active",
            "startDate": {"$gte": previous_30_days_start, "$lt": thirty_days_ago}
        })
        package_growth = ((
                                      current_active_packages - previous_active_packages) / previous_active_packages) * 100 if previous_active_packages > 0 else 0

        return {
            "totalUsers": total_users,
            "totalBookings": total_bookings,
            "totalRevenue": total_revenue,
            "activePackages": active_packages,
            "userGrowth": round(user_growth, 1),
            "bookingGrowth": round(booking_growth, 1),
            "revenueGrowth": round(revenue_growth, 1),
            "packageGrowth": round(package_growth, 1)
        }
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get admin stats",
        )


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
