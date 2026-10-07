"""Prelude symbol owners; token forms live with their declarations."""
from py_compiler.syntax.symbol.ellipsis import Ellipsis
from py_compiler.syntax.symbol.range import Range
from py_compiler.syntax.symbol.left_shift_equal import LeftShiftEqual
from py_compiler.syntax.symbol.right_shift_equal import RightShiftEqual
from py_compiler.syntax.symbol.power_equal import PowerEqual
from py_compiler.syntax.symbol.floor_divide_equal import FloorDivideEqual
from py_compiler.syntax.symbol.power import Power
from py_compiler.syntax.symbol.floor_divide import FloorDivide
from py_compiler.syntax.symbol.less_equal import LessEqual
from py_compiler.syntax.symbol.greater_equal import GreaterEqual
from py_compiler.syntax.symbol.equality import Equality
from py_compiler.syntax.symbol.not_equal import NotEqual
from py_compiler.syntax.symbol.colon_equal import ColonEqual
from py_compiler.syntax.symbol.question_equal import QuestionEqual
from py_compiler.syntax.symbol.arrow import Arrow
from py_compiler.syntax.symbol.conversion import Conversion
from py_compiler.syntax.symbol.plus_equal import PlusEqual
from py_compiler.syntax.symbol.minus_equal import MinusEqual
from py_compiler.syntax.symbol.star_equal import StarEqual
from py_compiler.syntax.symbol.slash_equal import SlashEqual
from py_compiler.syntax.symbol.percent_equal import PercentEqual
from py_compiler.syntax.symbol.ampersand_equal import AmpersandEqual
from py_compiler.syntax.symbol.pipe_equal import PipeEqual
from py_compiler.syntax.symbol.caret_equal import CaretEqual
from py_compiler.syntax.symbol.left_shift import LeftShift
from py_compiler.syntax.symbol.right_shift import RightShift
from py_compiler.syntax.symbol.left_paren import LeftParen
from py_compiler.syntax.symbol.right_paren import RightParen
from py_compiler.syntax.symbol.left_bracket import LeftBracket
from py_compiler.syntax.symbol.right_bracket import RightBracket
from py_compiler.syntax.symbol.left_brace import LeftBrace
from py_compiler.syntax.symbol.right_brace import RightBrace
from py_compiler.syntax.symbol.colon import Colon
from py_compiler.syntax.symbol.comma import Comma
from py_compiler.syntax.symbol.dot import Dot
from py_compiler.syntax.symbol.plus import Plus
from py_compiler.syntax.symbol.minus import Minus
from py_compiler.syntax.symbol.star import Star
from py_compiler.syntax.symbol.slash import Slash
from py_compiler.syntax.symbol.percent import Percent
from py_compiler.syntax.symbol.equal import Equal
from py_compiler.syntax.symbol.less import Less
from py_compiler.syntax.symbol.greater import Greater
from py_compiler.syntax.symbol.ampersand import Ampersand
from py_compiler.syntax.symbol.pipe import Pipe
from py_compiler.syntax.symbol.caret import Caret
from py_compiler.syntax.symbol.tilde import Tilde
from py_compiler.syntax.symbol.bang import Bang
from py_compiler.syntax.symbol.at import At
from py_compiler.syntax.symbol.semicolon import Semicolon


SYMBOLS = (
    Ellipsis(),
    Range(),
    LeftShiftEqual(),
    RightShiftEqual(),
    PowerEqual(),
    FloorDivideEqual(),
    Power(),
    FloorDivide(),
    LessEqual(),
    GreaterEqual(),
    Equality(),
    NotEqual(),
    ColonEqual(),
    QuestionEqual(),
    Arrow(),
    Conversion(),
    PlusEqual(),
    MinusEqual(),
    StarEqual(),
    SlashEqual(),
    PercentEqual(),
    AmpersandEqual(),
    PipeEqual(),
    CaretEqual(),
    LeftShift(),
    RightShift(),
    LeftParen(),
    RightParen(),
    LeftBracket(),
    RightBracket(),
    LeftBrace(),
    RightBrace(),
    Colon(),
    Comma(),
    Dot(),
    Plus(),
    Minus(),
    Star(),
    Slash(),
    Percent(),
    Equal(),
    Less(),
    Greater(),
    Ampersand(),
    Pipe(),
    Caret(),
    Tilde(),
    Bang(),
    At(),
    Semicolon(),
)
