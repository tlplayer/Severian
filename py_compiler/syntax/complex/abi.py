"""SIP-0039: target-selected scalar C ABI; no expression compilation here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CAbi:
    target: str
    pointer_bits: int = 64

    def __post_init__(self):
        if self.architecture not in ('x86_64','aarch64','arm64') or self.pointer_bits != 64:
            raise ValueError('C ABI requires a supported 64-bit x86_64 or aarch64 target')
        if self.system not in ('linux','darwin','windows'):
            raise ValueError('unsupported C ABI operating system: ' + self.target)

    @property
    def architecture(self): return self.target.split('-')[0]

    @property
    def system(self):
        parts=self.target.split('-')
        if any(p in ('windows','win32') for p in parts): return 'windows'
        if any(p.startswith(('darwin','macos')) for p in parts): return 'darwin'
        if 'linux' in parts: return 'linux'
        return ''

    @property
    def long_bits(self): return 32 if self.system=='windows' else 64

    def validate_type(self,type_,*,result=False):
        if getattr(type_,'no_result',False) and result: return
        if type_.family=='bool': return
        if type_.family=='integer' and type_.bits in (8,16,32,64): return
        if type_.family=='float' and type_.mlir in ('f32','f64'): return
        if type_.family=='pointer' and type_.mlir=='!llvm.ptr': return
        if type_.family=='string' and not result: return
        raise ValueError('unsupported C ABI '+('result' if result else 'parameter')+' type: '+type_.name)

    def extension(self,type_):
        extends=self.system!='windows' and (self.architecture=='x86_64' or self.system=='darwin')
        if extends and type_.family=='bool': return 'llvm.zeroext'
        if extends and type_.family=='integer' and type_.bits<32:
            return 'llvm.signext' if type_.signed else 'llvm.zeroext'
        return ''

    def attributes(self,type_):
        extension=self.extension(type_)
        return '{'+extension+'}' if extension else '{}'


import unittest


class CAbiTests(unittest.TestCase):
    def test_target_layout_and_extensions(self):
        from py_compiler.syntax.recognition import Syntax
        types=Syntax().types
        for triple,long_bits,extends in (
            ('x86_64-unknown-linux-gnu',64,True),('aarch64-unknown-linux-gnu',64,False),
            ('x86_64-pc-windows-msvc',32,False),('aarch64-pc-windows-msvc',32,False),
            ('aarch64-apple-darwin',64,True)):
            with self.subTest(target=triple):
                abi=CAbi(triple)
                self.assertEqual(abi.long_bits,long_bits)
                self.assertEqual(abi.extension(types['i8']),'llvm.signext' if extends else '')
                self.assertEqual(abi.extension(types['u16']),'llvm.zeroext' if extends else '')
                self.assertEqual(abi.extension(types['i32']),'')

    def test_unsupported_representations_do_not_guess(self):
        from py_compiler.syntax.recognition import Syntax
        abi=CAbi('x86_64-unknown-linux-gnu'); types=Syntax().types
        for name in ('i128','f16','f128','absent'):
            with self.subTest(type=name), self.assertRaisesRegex(ValueError,'unsupported C ABI'):
                abi.validate_type(types[name])
        abi.validate_type(types['absent'],result=True)
        with self.assertRaisesRegex(ValueError,'64-bit'): CAbi('i686-unknown-linux-gnu',32)
        with self.assertRaisesRegex(ValueError,'operating system'): CAbi('aarch64-unknown-none')


class AbiLibraryTests(unittest.TestCase):
    def test_xxi_consumes_the_published_abi_contract(self):
        from pathlib import Path
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        root=Path(__file__).resolve().parents[3]
        source=root/'library/system/abi/src/lib.sev'
        producer=compile_source(str(source),source.read_text(),Syntax(include_tests=False))
        self.assertFalse(producer.diagnostics,str(producer.diagnostics))
        declarations=producer.program.declarations
        types=[{'name':d.name,'fields':list(d.fields),'defaults':list(d.defaults),
                'traits':list(d.traits),'memory-defaults':list(d.memory_defaults)}
               for d in declarations if d.kind=='class']
        exports=[]
        for body in producer.program.bodies:
            if body.declaration!='c_target': continue
            parameters=body.bindings[:len(body.blocks[0].parameters)]
            exports.append({'name':body.declaration,'symbol':body.name,
                            'parameters':[{'name':b.name,'type':b.type.name,'ownership':b.ownership} for b in parameters],
                            'result':body.result_type.name,'result-sources':list(body.result_sources)})
        package={'package':{'name':'system.abi'},'objects':['abi.o'],'exports':exports,
                 'types':types,'templates':producer.templates}
        source=root/'library/system/xxi/src/lib.sev'
        consumer=compile_source(str(source),source.read_text(),Syntax(),c_packages={'abi':package})
        self.assertFalse(consumer.diagnostics,str(consumer.diagnostics))
