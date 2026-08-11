package mr

import (
	"log"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"sync"
	"time"
)

/*
The master keeps several data structures. For each map
task and reduce task, it stores the state (idle, in-progress,
or completed), and the identity of the worker machine
(for non-idle tasks).
The master is the conduit through which the location
of intermediate file regions is propagated from map tasks
to reduce tasks. Therefore, for each completed map task,
the master stores the locations and sizes of the R inter-
mediate file regions produced by the map task. Updates
to this location and size information are received as map
tasks are completed. The information is pushed incre-
mentally to workers that have in-progress reduce tasks.
*/
type TaskState int

const (
	Idle TaskState = iota
	InProgress
	Finished
)

type Task struct {
	Id        int
	File      string
	State     TaskState
	StartTime time.Time
}

type Phase int

const (
	MapPhase Phase = iota
	ReducePhase
	DonePhase
)

type Coordinator struct {
	mu sync.Mutex

	MapTasks    []Task
	ReduceTasks []Task

	Phase   Phase
	NReduce int
}

// Your code here -- RPC handlers for the worker to call.

func (c *Coordinator) RequestTask(args *RequestTaskArgs, reply *RequestTaskReply) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.checkTimeouts()

	switch c.Phase {
	case MapPhase:
		for i := range c.MapTasks {
			task := &c.MapTasks[i]

			if task.State == Idle {
				task.State = InProgress
				task.StartTime = time.Now()

				reply.Type = MapTask
				reply.TaskID = task.Id
				reply.File = task.File
				reply.NReduce = c.NReduce
				return nil
			}
		}

		if c.allMapsFinished() {
			c.Phase = ReducePhase
		}

		reply.Type = WaitTask
		return nil

	case ReducePhase:
		for i := range c.ReduceTasks {
			task := &c.ReduceTasks[i]

			if task.State == Idle {
				task.State = InProgress
				task.StartTime = time.Now()

				reply.Type = ReduceTask
				reply.TaskID = task.Id
				reply.NMap = len(c.MapTasks)
				return nil
			}
		}

		if c.allReducesFinished() {
			c.Phase = DonePhase
			reply.Type = ExitTask
			return nil
		}

		reply.Type = WaitTask
		return nil

	case DonePhase:
		reply.Type = ExitTask
		return nil
	}

	return nil
}

func (c *Coordinator) CompleteTask(args *CompleteTaskArgs, reply *CompleteTaskReply) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	switch args.Type {
	case MapTask:
		if args.TaskID >= 0 && args.TaskID < len(c.MapTasks) {
			c.MapTasks[args.TaskID].State = Finished
		}

		if c.allMapsFinished() {
			c.Phase = ReducePhase
		}

	case ReduceTask:
		if args.TaskID >= 0 && args.TaskID < len(c.ReduceTasks) {
			c.ReduceTasks[args.TaskID].State = Finished
		}

		if c.allReducesFinished() {
			c.Phase = DonePhase
		}
	}

	return nil
}

const taskTimeout = 10 * time.Second

func (c *Coordinator) checkTimeouts() {
	switch c.Phase {
	case MapPhase:
		for i := range c.MapTasks {
			task := &c.MapTasks[i]
			if task.State == InProgress &&
				time.Since(task.StartTime) > taskTimeout {

				task.State = Idle
				task.StartTime = time.Time{}
			}
		}
	case ReducePhase:
		for i := range c.ReduceTasks {
			task := &c.ReduceTasks[i]
			if task.State == InProgress &&
				time.Since(task.StartTime) > taskTimeout {

				task.State = Idle
				task.StartTime = time.Time{}
			}
		}
	default:
		return
	}
}

func (c *Coordinator) allMapsFinished() bool {
	for _, task := range c.MapTasks {
		if task.State != Finished {
			return false
		}
	}
	return true
}

func (c *Coordinator) allReducesFinished() bool {
	for _, task := range c.ReduceTasks {
		if task.State != Finished {
			return false
		}
	}
	return true
}

// start a thread that listens for RPCs from worker.go
func (c *Coordinator) server() {
	rpc.Register(c)
	rpc.HandleHTTP()
	//l, e := net.Listen("tcp", ":1234")
	sockname := coordinatorSock()
	os.Remove(sockname)
	l, e := net.Listen("unix", sockname)
	if e != nil {
		log.Fatal("listen error:", e)
	}
	go http.Serve(l, nil)
}

// main/mrcoordinator.go calls Done() periodically to find out
// if the entire job has finished.
func (c *Coordinator) Done() bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.Phase == DonePhase
}

// create a Coordinator.
// main/mrcoordinator.go calls this function.
// nReduce is the number of reduce tasks to use.
func MakeCoordinator(files []string, nReduce int) *Coordinator {
	c := Coordinator{
		Phase:   MapPhase,
		NReduce: nReduce,
	}

	// Create map tasks
	for i, file := range files {
		c.MapTasks = append(c.MapTasks, Task{
			Id:    i,
			File:  file,
			State: Idle,
		})
	}

	// Create reduce tasks
	for i := 0; i < nReduce; i++ {
		c.ReduceTasks = append(c.ReduceTasks, Task{
			Id:    i,
			State: Idle,
		})
	}

	c.server()

	return &c
}
