from typing import List

from fastapi import APIRouter, HTTPException

from anonymous_board.adapter.input.web.request.create_anonymous_board_request import CreateAnonymousBoardRequest
from anonymous_board.adapter.input.web.response.anonymous_board_response import AnonymousBoardResponse
from anonymous_board.application.usecase.anonymous_board_usecase import AnonymousBoardUseCase
from anonymous_board.infrastructure.repository.anonymous_board_repository_impl import AnonymousBoardRepositoryImpl

anonymous_board_router = APIRouter()
usecase = AnonymousBoardUseCase.getInstance()
# usecase = AnonymousBoardUseCase(AnonymousBoardRepositoryImpl())
# 이렇게 구성하면 이점이 무엇인가?
# anonymouse_board_usecase
# account_usecase 같이 여러 도메인들과 협력할 수 있는 구성의 경우
# 유일성을 보장해주는 것이 사용하기 유리한 측면이 있음.
# 그냥 무지성으로 getInstance() 호출하면 일단 유일한 객체라는 것이 보장됨.
# @RequiredArgsConstructor 입력 이후
# final private AccountService accountService 하는것과 같음
# @Autowired private AccountService accountService 하는것과 같음
# 결국 getInstance()로 위의 작업을 대신한다 생각하면 됩니다.
# 되도록 무지성으로 코딩할 수 있어야 생각할 요소가 상대적으로 적어지니까요.
# 도메인 구성 예쁘게 하는 것만 해도 머리 아픈데 저것까지 신경 쓰면 더 아픕니다.
# 최대한 문제를 단순화 시키는 것이 이득이기 때문에 위와 같은 선택을 하였습니다.

@anonymous_board_router.post("/create", response_model=AnonymousBoardResponse)
def create_board(request: CreateAnonymousBoardRequest):
    board = usecase.create_board(request.title, request.content)
    return AnonymousBoardResponse(
        id=board.id,
        title=board.title,
        content=board.content,
        created_at=board.created_at,
        updated_at=board.updated_at,
    )

@anonymous_board_router.get("/list", response_model=List[AnonymousBoardResponse])
def list_boards():
    boards = usecase.list_boards()
    return [
        AnonymousBoardResponse(
            id=b.id,
            title=b.title,
            content=b.content,
            created_at=b.created_at,
            updated_at=b.updated_at,
        ) for b in boards
    ]

@anonymous_board_router.get("/read/{board_id}", response_model=AnonymousBoardResponse)
def get_board(board_id: int):
    board = usecase.get_board(board_id)
    if not board:
        raise HTTPException(status_code=404, detail="Board not found")
    return AnonymousBoardResponse(
        id=board.id,
        title=board.title,
        content=board.content,
        created_at=board.created_at,
        updated_at=board.updated_at,
    )

@anonymous_board_router.delete("/delete/{board_id}")
def delete_board(board_id: int):
    success = usecase.delete_board(board_id)
    if not success:
        raise HTTPException(status_code=404, detail="Board not found")
    return {"message": "Deleted successfully"}
