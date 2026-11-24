"""
FitLoop Database Models and Persistence Layer
==============================================
SQLAlchemy ORM models and repository layer for FitLoop:
1. User: User identity and timestamps.
2. WardrobeScan: Ingestion scans ('closet', 'flat_lay', 'single_garment').
3. WardrobeItem: Detected garment items associated with user and scan.
4. DuplicateCheck: History of candidate duplicate evaluations.
5. WardrobeRepository: Data-access layer with strict user data isolation and
   seamless interop with DuplicateDetector.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    selectinload,
    sessionmaker,
)


# Enforce foreign key constraints in SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        pass


class Base(DeclarativeBase):
    """Base declarative class for FitLoop database models."""
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    scans: Mapped[List[WardrobeScan]] = relationship(
        "WardrobeScan",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    items: Mapped[List[WardrobeItem]] = relationship(
        "WardrobeItem",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    duplicate_checks: Mapped[List[DuplicateCheck]] = relationship(
        "DuplicateCheck",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, name='{self.name}', email='{self.email}')>"


class WardrobeScan(Base):
    __tablename__ = "wardrobe_scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_url: Mapped[str] = mapped_column(String(500), nullable=False)
    input_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="single_garment",
    )  # 'closet', 'flat_lay', 'single_garment'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="scans")
    items: Mapped[List[WardrobeItem]] = relationship(
        "WardrobeItem",
        back_populates="scan",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<WardrobeScan(id={self.id}, user_id={self.user_id}, type='{self.input_type}')>"


class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("wardrobe_scans.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    subcategory: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    colour_name: Mapped[str] = mapped_column(String(60), nullable=False)
    colour_hex: Mapped[str] = mapped_column(String(10), nullable=False)
    secondary_colour: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    secondary_colour_hex: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    image_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    pattern: Mapped[Optional[str]] = mapped_column(String(50), default="solid", nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="items")
    scan: Mapped[Optional[WardrobeScan]] = relationship("WardrobeScan", back_populates="items")
    matched_checks: Mapped[List[DuplicateCheck]] = relationship(
        "DuplicateCheck",
        back_populates="matched_item",
    )
    embedding_record: Mapped[Optional[ItemEmbedding]] = relationship(
        "ItemEmbedding",
        back_populates="item",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def to_detector_dict(self) -> Dict[str, Any]:
        """Convert to dict formatted for DuplicateDetector."""
        sec_color = None
        if self.secondary_colour:
            sec_color = {
                "name": self.secondary_colour,
                "hex": self.secondary_colour_hex or "#000000",
            }

        emb = None
        if self.embedding_record and self.embedding_record.embedding_json:
            try:
                emb = json.loads(self.embedding_record.embedding_json)
            except Exception:
                emb = None

        return {
            "id": f"item_{self.id}",
            "db_id": self.id,
            "name": f"{self.colour_name} {self.category}".title(),
            "category": self.category,
            "subcategory": self.subcategory,
            "dominant_color": {
                "name": self.colour_name,
                "hex": self.colour_hex,
            },
            "secondary_color": sec_color,
            "pattern": self.pattern or "solid",
            "confidence": self.confidence,
            "image_url": self.image_url,
            "embedding": emb,
            "purchase_date": self.created_at.strftime("%b %d, %Y"),
        }

    def __repr__(self) -> str:
        return f"<WardrobeItem(id={self.id}, user={self.user_id}, category='{self.category}', color='{self.colour_name}')>"


class ItemEmbedding(Base):
    __tablename__ = "item_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("wardrobe_items.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), default="mobilenet_v3_small", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    item: Mapped[WardrobeItem] = relationship("WardrobeItem", back_populates="embedding_record")

    def __repr__(self) -> str:
        return f"<ItemEmbedding(id={self.id}, item_id={self.item_id}, model='{self.model_version}')>"


class DuplicateCheck(Base):
    __tablename__ = "duplicate_checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_category: Mapped[str] = mapped_column(String(100), nullable=False)
    candidate_color: Mapped[str] = mapped_column(String(60), nullable=False)
    matched_item_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("wardrobe_items.id", ondelete="SET NULL"),
        nullable=True,
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    verdict: Mapped[str] = mapped_column(String(30), nullable=False)  # 'DUPLICATE', 'SIMILAR', 'UNIQUE'
    recommendation: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="duplicate_checks")
    matched_item: Mapped[Optional[WardrobeItem]] = relationship(
        "WardrobeItem",
        back_populates="matched_checks",
    )

    def __repr__(self) -> str:
        return f"<DuplicateCheck(id={self.id}, user={self.user_id}, verdict='{self.verdict}', score={self.similarity_score:.2f})>"


class WardrobeRepository:
    """
    Repository interface managing user wardrobe operations, persistence, and isolation.
    """

    def __init__(self, session_or_engine: Union[Session, Engine, str]) -> None:
        if isinstance(session_or_engine, str):
            self.engine = create_engine(session_or_engine)
            Base.metadata.create_all(self.engine)
            self._session_factory = sessionmaker(bind=self.engine)
            self.session = self._session_factory()
            self._owns_session = True
        elif isinstance(session_or_engine, Engine):
            self.engine = session_or_engine
            Base.metadata.create_all(self.engine)
            self._session_factory = sessionmaker(bind=self.engine)
            self.session = self._session_factory()
            self._owns_session = True
        elif isinstance(session_or_engine, Session):
            self.session = session_or_engine
            self._owns_session = False
        else:
            raise TypeError(f"Unsupported session/engine type: {type(session_or_engine)}")

    def close(self) -> None:
        """Close open database session and dispose connection pool if owned."""
        if getattr(self, "_owns_session", False):
            self.session.close()
            if hasattr(self, "engine"):
                self.engine.dispose()

    def create_user(self, name: str, email: str) -> User:
        """Create and persist a new user."""
        user = User(name=name.strip(), email=email.strip().lower())
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Retrieve user by primary key ID."""
        return self.session.get(User, user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Retrieve user by unique email."""
        stmt = select(User).where(User.email == email.strip().lower())
        return self.session.scalars(stmt).first()

    def add_scan_and_items(
        self,
        user_id: int,
        scan_metadata: Dict[str, Any],
        detected_items: List[Dict[str, Any]],
    ) -> Tuple[WardrobeScan, List[WardrobeItem]]:
        """
        Record a wardrobe scan along with all detected garment items under a single transaction.

        Args:
            user_id: Foreign key ID of the user owning the scan.
            scan_metadata: Dict with 'image_url', optional 'input_type'.
            detected_items: List of dicts representing detected garments:
                {
                    'category': 'short sleeve top',
                    'subcategory': 't-shirt',
                    'colour_name': 'Navy Blue',
                    'colour_hex': '#1B263B',
                    'secondary_colour': 'Olive Green',
                    'secondary_colour_hex': '#556B2F',
                    'confidence': 0.95,
                    'image_url': 'crops/garment_01.jpg',
                    'pattern': 'solid',
                }

        Returns:
            Tuple of (persisted WardrobeScan, list of persisted WardrobeItem instances).
        """
        user = self.get_user_by_id(user_id)
        if user is None:
            raise ValueError(f"User with ID {user_id} does not exist.")

        # Create scan
        scan = WardrobeScan(
            user_id=user_id,
            image_url=scan_metadata.get("image_url", ""),
            input_type=scan_metadata.get("input_type", "single_garment"),
        )
        self.session.add(scan)
        self.session.flush()  # Populates scan.id

        created_items: List[WardrobeItem] = []

        for item_data in detected_items:
            # Handle color inputs that may be dict or string
            col_name = item_data.get("colour_name")
            col_hex = item_data.get("colour_hex")
            if not col_name and isinstance(item_data.get("dominant_color"), dict):
                col_name = item_data["dominant_color"].get("name")
                col_hex = item_data["dominant_color"].get("hex")

            sec_name = item_data.get("secondary_colour")
            sec_hex = item_data.get("secondary_colour_hex")
            if not sec_name and isinstance(item_data.get("secondary_color"), dict):
                sec_name = item_data["secondary_color"].get("name")
                sec_hex = item_data["secondary_color"].get("hex")

            item = WardrobeItem(
                scan_id=scan.id,
                user_id=user_id,
                category=item_data["category"],
                subcategory=item_data.get("subcategory"),
                colour_name=col_name or "Unknown",
                colour_hex=col_hex or "#000000",
                secondary_colour=sec_name,
                secondary_colour_hex=sec_hex,
                confidence=float(item_data.get("confidence", 1.0)),
                image_url=item_data.get("image_url"),
                pattern=item_data.get("pattern", "solid"),
            )

            raw_emb = item_data.get("embedding")
            if raw_emb is not None:
                if hasattr(raw_emb, "tolist"):
                    raw_emb = raw_emb.tolist()
                model_ver = str(item_data.get("model_version", "mobilenet_v3_small"))
                item.embedding_record = ItemEmbedding(
                    embedding_json=json.dumps(raw_emb),
                    model_version=model_ver,
                )

            self.session.add(item)
            created_items.append(item)

        self.session.commit()
        self.session.refresh(scan)
        for it in created_items:
            self.session.refresh(it)

        return scan, created_items

    def get_user_wardrobe(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Query all active wardrobe items for a given user, formatted for DuplicateDetector.
        Guarantees strict user data isolation.
        """
        stmt = (
            select(WardrobeItem)
            .options(selectinload(WardrobeItem.embedding_record))
            .where(WardrobeItem.user_id == user_id)
            .order_by(WardrobeItem.created_at.desc())
        )
        items = self.session.scalars(stmt).all()
        return [item.to_detector_dict() for item in items]

    def log_duplicate_check(
        self,
        user_id: int,
        check_results: Dict[str, Any],
        candidate_category: Optional[str] = None,
        candidate_color: Optional[str] = None,
    ) -> DuplicateCheck:
        """
        Persist a record of a duplicate check evaluation.

        Args:
            user_id: The checking user's ID.
            check_results: The structured dict returned by DuplicateDetector.check_duplicate().
            candidate_category: Optional explicit candidate category.
            candidate_color: Optional explicit candidate color.
        """
        matched_item = check_results.get("matched_item")
        matched_id = None
        if isinstance(matched_item, dict):
            matched_id = matched_item.get("db_id")

        cat = candidate_category or "Unknown"
        col = candidate_color or "Unknown"

        log_entry = DuplicateCheck(
            user_id=user_id,
            candidate_category=cat,
            candidate_color=col,
            matched_item_id=matched_id,
            similarity_score=float(check_results.get("similarity_score", 0.0)),
            verdict=str(check_results.get("verdict", "UNIQUE")),
            recommendation=check_results.get("recommendation"),
        )
        self.session.add(log_entry)
        self.session.commit()
        self.session.refresh(log_entry)
        return log_entry

    def get_user_duplicate_checks(self, user_id: int) -> List[DuplicateCheck]:
        """Retrieve all duplicate check history records for a user."""
        stmt = (
            select(DuplicateCheck)
            .where(DuplicateCheck.user_id == user_id)
            .order_by(DuplicateCheck.created_at.desc())
        )
        return list(self.session.scalars(stmt).all())

    def delete_wardrobe_item(self, item_id: int, user_id: int) -> bool:
        """Delete a wardrobe item and its associated records if belonging to user_id."""
        stmt = select(WardrobeItem).where(WardrobeItem.id == item_id, WardrobeItem.user_id == user_id)
        item = self.session.scalars(stmt).first()
        if not item:
            return False
        # Nullify any foreign key references in duplicate checks
        self.session.execute(
            update(DuplicateCheck).where(DuplicateCheck.matched_item_id == item_id).values(matched_item_id=None)
        )
        self.session.delete(item)
        self.session.commit()
        return True

    def update_wardrobe_item(self, item_id: int, user_id: int, updates: Dict[str, Any]) -> Optional[WardrobeItem]:
        """Update properties (category, colour_name, colour_hex, etc.) of a wardrobe item."""
        stmt = select(WardrobeItem).where(WardrobeItem.id == item_id, WardrobeItem.user_id == user_id)
        item = self.session.scalars(stmt).first()
        if not item:
            return None
        if "category" in updates and updates["category"]:
            item.category = updates["category"]
            item.subcategory = updates["category"]
        if "colour_hex" in updates and updates["colour_hex"]:
            item.colour_hex = updates["colour_hex"]
        if "colour_name" in updates and updates["colour_name"]:
            item.colour_name = updates["colour_name"]
        self.session.commit()
        self.session.refresh(item)
        return item

