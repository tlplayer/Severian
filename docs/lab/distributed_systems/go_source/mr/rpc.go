package mr

//
// RPC definitions.
//
//

import (
	"os"
	"strconv"
)

type TaskType int

const (
	MapTask TaskType = iota
	ReduceTask
	WaitTask
	ExitTask
)

type RequestTaskArgs struct{}

type RequestTaskReply struct {
	Type    TaskType
	TaskID  int
	File    string
	NReduce int
	NMap    int
}

type CompleteTaskArgs struct {
	Type   TaskType
	TaskID int
}

type CompleteTaskReply struct{}

// Cook up a unique-ish UNIX-domain socket name
// in /var/tmp, for the coordinator.
// Can't use the current directory since
// Athena AFS doesn't support UNIX-domain sockets.
func coordinatorSock() string {
	s := "/var/tmp/5840-mr-"
	s += strconv.Itoa(os.Getuid())
	return s
}
