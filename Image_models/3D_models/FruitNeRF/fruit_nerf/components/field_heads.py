from typing import Optional
from nerfstudio.field_components.field_heads import FieldHead, FieldHeadNames

class SemanticFieldHead(FieldHead):
    def __init__(self, num_classes: int, in_dim: Optional[int] = None, activation=None) -> None:
        super().__init__(in_dim=in_dim, out_dim=num_classes, field_head_name=FieldHeadNames.SEMANTICS, activation=activation)