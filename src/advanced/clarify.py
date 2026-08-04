"""Làm rõ câu hỏi TRƯỚC khi truy xuất.

Vấn đề: "trả hàng" là một CHỦ ĐỀ, không phải một câu hỏi.

Nếu đem thẳng chuỗi đó đi embedding, vector thu được nằm ở "vùng trung tâm"
của toàn bộ chương trả hàng — gần đều với điều kiện, thời hạn, quy trình, chi
phí, hạn mức. Kết quả là top-5 gom mỗi thứ một ít, không cái nào trúng ý người
hỏi. Đây là lỗi không thể sửa bằng cách chỉnh retrieval, vì thông tin còn
thiếu nằm ở phía NGƯỜI DÙNG chứ không nằm trong tài liệu.

Cách xử lý: nhận diện câu chưa đủ cụ thể, hỏi lại 1 lượt với các lựa chọn bấm
được, rồi mới truy xuất bằng câu hỏi đã hoàn chỉnh.

Nguyên tắc để không phiền người dùng:
    1. Chỉ hỏi khi thật sự mơ hồ. Câu đã có từ để hỏi ("bao lâu", "thế nào")
       và nêu rõ khía cạnh thì cho đi thẳng.
    2. Tối đa MỘT lượt hỏi lại. Không truy vấn ngược liên tục.
    3. Chỉ hỏi những gì thay đổi được câu trả lời — trọng tâm là vai trò
       người mua/người bán, vì hai bên có quy định khác hẳn nhau.
"""

import re
from dataclasses import dataclass, field

# Từ để hỏi — có mặt nghĩa là người dùng đã nêu rõ họ muốn biết gì.
_QUESTION_MARKERS = [
    r"\?",
    r"\b(bao\s+lâu|bao\s+nhiêu|mấy\s+ngày|khi\s+nào|lúc\s+nào)\b",
    r"\b(thế\s+nào|như\s+thế\s+nào|làm\s+sao|làm\s+thế\s+nào|ra\s+sao|cách\s+nào)\b",
    r"\b(là\s+gì|gồm\s+những\s+gì|những\s+gì|cái\s+gì)\b",
    r"\b(ai|đâu|ở\s+đâu|tại\s+sao|vì\s+sao|có\s+được|được\s+không|có\s+phải)\b",
    r"\b(điều\s+kiện|quy\s+trình|thủ\s+tục|hồ\s+sơ|thời\s+hạn|hạn\s+mức|chi\s+phí|mức\s+phí)\b",
    r"\b(cho\s+tôi\s+biết|giải\s+thích|hướng\s+dẫn|liệt\s+kê|so\s+sánh|tóm\s+tắt)\b",
]

# Người dùng đã tự nêu vai trò → không cần hỏi lại.
_ROLE_MARKERS = {
    "buyer": [r"\b(người\s+mua|khách\s+hàng|tôi\s+mua|tôi\s+là\s+người\s+mua|buyer)\b"],
    "seller": [r"\b(người\s+bán|nhà\s+bán|chủ\s+shop|gian\s+hàng|tôi\s+bán|seller|shop\s+của\s+tôi)\b"],
}


@dataclass
class ClarifyQuestion:
    key: str
    text: str
    options: list[str] = field(default_factory=list)


@dataclass
class ClarifyResult:
    needed: bool
    topic: str = ""
    topic_label: str = ""
    reason: str = ""
    questions: list[ClarifyQuestion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "needed": self.needed,
            "topic": self.topic,
            "topic_label": self.topic_label,
            "reason": self.reason,
            "questions": [
                {"key": q.key, "text": q.text, "options": q.options} for q in self.questions
            ],
        }


# Mỗi chủ đề: từ khoá nhận diện, có phụ thuộc vai trò không, và các khía cạnh
# mà người dùng có thể muốn biết. Danh sách khía cạnh bám sát cấu trúc thật của
# bộ tài liệu, nên chọn xong là truy xuất trúng ngay.
_TOPICS = {
    "return_refund": {
        "label": "Trả hàng / Hoàn tiền",
        "patterns": [r"\b(trả\s+hàng|hoàn\s+tiền|đổi\s+trả|trả\s+lại|hoàn\s+trả|refund|return)\b"],
        "role_sensitive": True,
        "aspects": [
            "Điều kiện được trả hàng",
            "Thời hạn gửi yêu cầu",
            "Quy trình thực hiện",
            "Ai chịu phí vận chuyển",
            "Hình thức nhận tiền hoàn",
            "Trả hàng COM và hạn mức",
        ],
    },
    "payment": {
        "label": "Thanh toán",
        "patterns": [r"\b(thanh\s+toán|phương\s+thức\s+tt|ví|thẻ|cod|chuyển\s+khoản|spaylater|napas|payment)\b"],
        "role_sensitive": False,
        "aspects": [
            "Các phương thức được hỗ trợ",
            "Cách đổi phương thức thanh toán",
            "Sự cố khi thanh toán",
            "Điều kiện để được hoàn tiền về ví",
        ],
    },
    "shipping": {
        "label": "Vận chuyển",
        "patterns": [r"\b(vận\s+chuyển|giao\s+hàng|phí\s+ship|đơn\s+vị\s+vận\s+chuyển|giao\s+nhận|shipping)\b"],
        "role_sensitive": True,
        "aspects": [
            "Ai chịu phí vận chuyển",
            "Các hình thức gửi trả hàng",
            "Quy định đóng gói",
            "Giao hàng không thành công",
        ],
    },
    "privacy": {
        "label": "Bảo mật & Quyền riêng tư",
        "patterns": [r"\b(bảo\s+mật|riêng\s+tư|dữ\s+liệu\s+cá\s+nhân|thông\s+tin\s+cá\s+nhân|privacy)\b"],
        "role_sensitive": False,
        "aspects": [
            "Dữ liệu nào được thu thập",
            "Dữ liệu được dùng làm gì",
            "Quyền của người dùng với dữ liệu",
            "Xử lý khi có giao dịch lạ",
        ],
    },
    "seller_rules": {
        "label": "Quy định người bán",
        "patterns": [r"\b(người\s+bán|đăng\s+bán|gian\s+hàng|shop|niêm\s+yết|sản\s+phẩm\s+cấm|seller)\b"],
        "role_sensitive": False,
        "aspects": [
            "Sản phẩm bị cấm đăng bán",
            "Trách nhiệm khi khách trả hàng",
            "Thời hạn phản hồi yêu cầu",
            "Chi phí người bán phải chịu",
        ],
    },
    "order": {
        "label": "Đơn hàng",
        "patterns": [r"\b(đơn\s+hàng|theo\s+dõi\s+đơn|trạng\s+thái\s+đơn|mã\s+đơn|đặt\s+hàng)\b"],
        "role_sensitive": False,
        "aspects": [
            "Cách theo dõi đơn hàng",
            "Huỷ đơn hàng",
            "Đơn hàng giao không thành công",
            "Khiếu nại về đơn hàng",
        ],
    },
}


def _has(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE | re.UNICODE) for p in patterns)


def _detect_topic(text: str) -> tuple[str, dict] | tuple[str, None]:
    for topic_id, spec in _TOPICS.items():
        if _has(text, spec["patterns"]):
            return topic_id, spec
    return "", None


def _detect_role(text: str) -> str:
    for role, patterns in _ROLE_MARKERS.items():
        if _has(text, patterns):
            return role
    return ""


def _asked_recently(history: list[dict] | None) -> bool:
    """Đã hỏi lại ở lượt ngay trước chưa? Tránh hỏi vòng vo không dứt."""
    if not history:
        return False
    for turn in reversed(history):
        if turn.get("role") == "assistant":
            return bool(turn.get("clarify"))
    return False


def analyze(query: str, history: list[dict] | None = None) -> ClarifyResult:
    """Quyết định có cần hỏi lại trước khi truy xuất hay không."""
    text = re.sub(r"\s+", " ", str(query or "")).strip().lower()
    if not text:
        return ClarifyResult(False)

    # Đã hỏi lại một lần rồi thì thôi — người dùng vừa trả lời xong.
    if _asked_recently(history):
        return ClarifyResult(False, reason="Đã hỏi lại ở lượt trước.")

    topic_id, spec = _detect_topic(text)
    if not spec:
        # Không nhận ra chủ đề: để cổng bằng chứng phía sau xử lý, đừng hỏi mò.
        return ClarifyResult(False, reason="Không nhận diện được chủ đề cụ thể.")

    content_tokens = [t for t in re.findall(r"\w+", text, flags=re.UNICODE) if len(t) > 1]
    has_marker = _has(text, _QUESTION_MARKERS)
    role = _detect_role(text)

    # Câu đã nêu rõ muốn biết gì VÀ đủ dài → truy xuất luôn, không làm phiền.
    if has_marker and len(content_tokens) >= 5:
        return ClarifyResult(False, reason="Câu hỏi đã đủ cụ thể.")

    # Câu ngắn, không có từ để hỏi → chỉ là nêu chủ đề.
    if not has_marker or len(content_tokens) < 5:
        questions = []
        if spec["role_sensitive"] and not role:
            questions.append(ClarifyQuestion(
                key="role",
                text="Bạn đang hỏi với tư cách nào?",
                options=["Người mua", "Người bán"],
            ))
        questions.append(ClarifyQuestion(
            key="aspect",
            text=f"Bạn muốn biết điều gì về {spec['label'].lower()}?",
            options=list(spec["aspects"]),
        ))
        return ClarifyResult(
            needed=True,
            topic=topic_id,
            topic_label=spec["label"],
            reason=(
                f"“{query.strip()}” mới là chủ đề, chưa phải câu hỏi. "
                "Truy xuất ngay sẽ trả về mỗi thứ một ít mà không trúng ý."
            ),
            questions=questions,
        )

    return ClarifyResult(False, reason="Câu hỏi đã đủ cụ thể.")


def build_refined_query(original: str, topic_label: str, answers: dict) -> str:
    """Ghép lựa chọn của người dùng thành một câu hỏi hoàn chỉnh.

    Câu ghép ra vừa là truy vấn cho retrieval, vừa là câu hỏi hiển thị lại cho
    người dùng, nên phải đọc trôi chảy chứ không được là chuỗi từ khoá ghép máy móc.
    """
    role_text = {"Người mua": "Với tư cách người mua, ", "Người bán": "Với tư cách người bán, "}
    prefix = role_text.get(answers.get("role", ""), "")
    aspect = answers.get("aspect", "").strip()
    topic = topic_label.lower()

    if not aspect:
        return f"{prefix}{original.strip()} được quy định thế nào?"

    # Có tiền tố vai trò thì chữ đầu của khía cạnh phải viết thường.
    if prefix:
        aspect = aspect[0].lower() + aspect[1:]

    return f"{prefix}{aspect} trong chính sách {topic} được quy định thế nào?"


def clarification_message(result: ClarifyResult) -> str:
    """Nội dung hiển thị khi hệ thống hỏi lại."""
    lines = [
        "Trước khi tra cứu, tôi cần làm rõ câu hỏi để tìm đúng điều khoản.",
        "",
        f"_{result.reason}_",
        "",
    ]
    for question in result.questions:
        lines.append(f"**{question.text}**")
        lines.append("")
    return "\n".join(lines).strip()
