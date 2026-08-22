from abc import ABC, abstractmethod


class Opcode(ABC):
    """
    Base class for A64 instructions.

    The three layers work as in the ARMv6 model: a decoder returns one of these classes,
    ``from_bitarray`` pulls the operand fields out of the encoding, and ``execute`` carries
    the semantics. Encodings that share semantics share an abstract subclass.

    ``from_bitarray`` returns None when the encoding is reserved or otherwise UNDEFINED,
    which the CPU model turns into an undefined instruction exception.
    """

    def __init__(self, instruction):
        self.instruction = instruction

    @staticmethod
    def from_bitarray(instr, processor):
        raise NotImplementedError()

    @abstractmethod
    def execute(self, processor):
        """
        Execute the opcode on the given processor.
        :param processor: Processor to run the opcode on.
        """
        raise NotImplementedError()

    def __repr__(self):
        return f'<{type(self).__name__} 0x{self.instruction:08X}>'
