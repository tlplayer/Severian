use super::*;

impl Parser<'_> {
    pub(super) fn parse_type(&mut self) -> Result<Type, ParseError> {
        let first = self.parse_named_type()?;
        if !self.at(&TokenKind::Pipe) {
            return Ok(first);
        }
        let start = first.span().start;
        let mut alternatives = vec![first];
        while self.take_simple(&TokenKind::Pipe).is_some() {
            alternatives.push(self.parse_named_type()?);
        }
        let end = alternatives.last().unwrap().span().end;
        Ok(Type::Union {
            span: Span::new(start, end),
            alternatives,
        })
    }

    pub(super) fn parse_named_type(&mut self) -> Result<Type, ParseError> {
        let name = self.expect_identifier("type name")?;
        let start = name.span.start;
        let mut segments = vec![name];
        while self.take_simple(&TokenKind::Dot).is_some() {
            segments.push(self.expect_identifier("type path segment")?);
        }
        let mut args = Vec::new();
        let mut end = segments.last().unwrap().span.end;
        if self.take_simple(&TokenKind::LeftBracket).is_some() {
            if !self.at(&TokenKind::RightBracket) {
                loop {
                    let tensor_dimension = segments
                        .first()
                        .is_some_and(|segment| segment.name == "Tensor")
                        && !args.is_empty();
                    if tensor_dimension && matches!(self.peek().kind, TokenKind::Integer(_)) {
                        let TokenKind::Integer(value) = self.peek().kind else {
                            unreachable!()
                        };
                        let token = self.advance().clone();
                        let size = u64::try_from(value)
                            .map_err(|_| self.error("tensor dimensions cannot be negative"))?;
                        args.push(TypeArg::Dimension {
                            span: token.span,
                            size,
                        });
                    } else {
                        let ty = self.parse_type()?;
                        args.push(TypeArg::Type {
                            span: ty.span(),
                            ty: Box::new(ty),
                        });
                    }
                    if self.take_simple(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
            }
            end = self.expect_simple(TokenKind::RightBracket, "`]`")?.span.end;
        }
        Ok(Type::Named(TypePath {
            span: Span::new(start, end),
            segments,
            args,
        }))
    }
}
